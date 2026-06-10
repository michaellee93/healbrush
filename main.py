import time
from pathlib import Path
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from dataloader import HealDataset
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision.models import vgg16, VGG16_Weights

vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().cuda()
for param in vgg.parameters():
    param.requires_grad = False


def perceptual_loss(output, target):
    return F.l1_loss(vgg(output), vgg(target))


# GAN fine-tune knobs. The generator is fully trained already — we resume its
# weights and drop its LR so the adversarial gradient nudges rather than wrecks
# the minimum it already found. The discriminator starts from scratch.
ADV_WEIGHT = 0.0
FINE_TUNE_LR = 5e-5

# DTD is ~5.6k texture crops vs. COCO's ~118k photos — sampled at its natural
# frequency it'd barely register. This multiplies how often a DTD sample is
# drawn relative to a COCO sample drawn at its own natural frequency, so the
# model sees meaningfully more texture variety without us duplicating files
# on disk. Keep it modest: too high and the model learns "fill with a generic
# texture swatch" instead of context-appropriate content.
DTD_OVERSAMPLE = 3.0


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = self.conv(x)
        down = self.pool(features)
        return features, down


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SelfAttention2d(nn.Module):
    """Self-attention over spatial positions, applied at the bottleneck where
    the feature map is small enough (16x16 at 256px input) for full
    pairwise attention to be cheap."""

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x + residual


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # input is 4 rgb + mask
        self.down1 = Down(4, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.bottleneck = ConvBlock(512, 512)
        self.attn = SelfAttention2d(512)

        # upps
        self.up4 = Up(512, 512, 512)
        self.up3 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up1 = Up(128, 64, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        x = self.bottleneck(x)
        x = self.attn(x)

        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return torch.sigmoid(self.out(x))


class Discriminator(nn.Module):
    """PatchGAN-style critic, conditioned on the hole context and mask.

    Judging "is this fill plausible *given this hole*" (rather than "is this a
    real photo") is what makes a conditional discriminator useful for inpainting
    specifically — an unconditional one can be satisfied by any sharp texture,
    which doesn't push the generator toward content that fits the surroundings.
    Outputs a patch grid of logits (no sigmoid — paired with hinge loss) so the
    signal is local texture quality, not a single whole-image verdict.
    """

    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, stride=2, norm=True):
            layers = [
                nn.utils.spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
                )
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            # input: pred/clean (3) + context (3) + mask (1) = 7 channels
            *block(7, 64, norm=False),    # 256 -> 128
            *block(64, 128),              # 128 -> 64
            *block(128, 256),             # 64  -> 32
            *block(256, 512, stride=1),   # 32  -> 31
            nn.utils.spectral_norm(
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            ),                            # 31  -> 30  (patch logits)
        )

    def forward(self, image, context, mask):
        x = torch.cat([image, context, mask], dim=1)
        return self.net(x)


def hinge_d_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()


def healing_loss(pred, target, mask):
    diff = (pred - target).abs()
    hole_loss = (diff * mask).sum() / (mask.sum() + 1e-8)
    valid_loss = (diff * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
    return hole_loss + 0.1 * valid_loss


def make_eval_batch(dataset, n=5):
    samples = [dataset[i] for i in range(n)]
    x = torch.stack([s[0] for s in samples]).cuda()
    clean = torch.stack([s[1] for s in samples]).cuda()
    mask = torch.stack([s[2] for s in samples]).cuda()
    return x, clean, mask


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    wandb.init(project="healbrush", resume="allow")

    out_dir = Path("./outputs/attn")
    out_dir.mkdir(exist_ok=True)

    model = torch.compile(UNet().cuda())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    disc = Discriminator().cuda()
    opt_d = torch.optim.AdamW(disc.parameters(), lr=4e-5, betas=(0.5, 0.999))

    coco_dataset = HealDataset("./data/coco", size=256)
    dtd_dataset = HealDataset("./data/dtd/images", size=256)
    dataset = ConcatDataset([coco_dataset, dtd_dataset])

    # Per-sample weight = (oversample factor) / (dataset size), so within each
    # source every sample is drawn at a uniform rate, and DTD samples overall
    # are drawn DTD_OVERSAMPLE times more often than their natural 5.6k/118k
    # share would give them.
    coco_weight = 1.0 / len(coco_dataset)
    dtd_weight = DTD_OVERSAMPLE / len(dtd_dataset)
    sample_weights = [coco_weight] * len(coco_dataset) + [dtd_weight] * len(dtd_dataset)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)

    loader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    print(f"coco={len(coco_dataset)} dtd={len(dtd_dataset)} total={len(dataset)}")
    print(len(loader))

    torch.manual_seed(42)
    eval_x, eval_clean, eval_mask = make_eval_batch(coco_dataset)
    torch.seed()

    step = 0
    start_epoch = 0
    ckpt_path = out_dir / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"resumed from checkpoint: step={step}, epoch={start_epoch}")

        if "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"])
            opt_d.load_state_dict(ckpt["opt_d"])
            print("resumed discriminator from checkpoint — continuing GAN fine-tune")
        else:
            # First time the discriminator appears: this checkpoint was trained
            # with reconstruction losses only. Drop the generator's LR so the
            # adversarial signal nudges it rather than knocking it out of the
            # minimum it already converged to; the discriminator starts fresh.
            for group in opt.param_groups:
                group["lr"] = FINE_TUNE_LR
            print(f"no discriminator in checkpoint — starting GAN fine-tune at lr={FINE_TUNE_LR}")

    t0 = time.time()
    model.train()
    for epoch in range(start_epoch, 20):
        for x, clean, mask in loader:
            x = x.cuda()
            clean = clean.cuda()
            mask = mask.cuda()
            context = x[:, :3]

            pred = model(x)

            # --- discriminator step: real vs. fake, conditioned on the hole ---
            d_loss = torch.zeros(())
            if ADV_WEIGHT > 0:
                opt_d.zero_grad()
                real_logits = disc(clean, context, mask)
                fake_logits = disc(pred.detach(), context, mask)
                d_loss = hinge_d_loss(real_logits, fake_logits)
                d_loss.backward()
                opt_d.step()

            # --- generator step: reconstruction + perceptual + adversarial ---
            opt.zero_grad()
            adv_loss = torch.zeros(())
            if ADV_WEIGHT > 0:
                adv_logits = disc(pred, context, mask)
                adv_loss = hinge_g_loss(adv_logits)
            loss = (
                healing_loss(pred, clean, mask)
                + 0.1 * perceptual_loss(pred, clean)
                + ADV_WEIGHT * adv_loss
            )
            loss.backward()
            opt.step()

            step += 1
            wandb.log(
                {"loss": loss.item(), "d_loss": d_loss.item(), "adv_loss": adv_loss.item()},
                step=step,
            )

            if step == 100 or step % 2000 == 0:
                elapsed = time.time() - t0
                print(
                    f"[{elapsed:8.1f}s] step {step}  "
                    f"loss {loss.item():.4f}  d_loss {d_loss.item():.4f}  adv {adv_loss.item():.4f}"
                )

                model.eval()
                with torch.no_grad():
                    # fixed eval grid
                    eval_pred = model(eval_x)
                    eval_final = eval_clean * (1 - eval_mask) + eval_pred * eval_mask
                    eval_grid = torch.cat(
                        [
                            eval_clean,
                            eval_x[:, :3],
                            eval_mask.repeat(1, 3, 1, 1),
                            eval_pred,
                            eval_final,
                        ],
                        dim=0,
                    )
                    save_image(eval_grid, out_dir / f"eval_{step:06d}.png", nrow=5)
                    wandb.log({"eval": wandb.Image(str(out_dir / f"eval_{step:06d}.png"))}, step=step)

                    # checkpoint
                    torch.save(
                        {
                            "step": step,
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "disc": disc.state_dict(),
                            "opt_d": opt_d.state_dict(),
                        },
                        out_dir / "checkpoint.pt",
                    )
                model.train()

    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "disc": disc.state_dict(),
            "opt_d": opt_d.state_dict(),
        },
        out_dir / "final.pt",
    )
    print(f"saved final.pt at step {step}")


if __name__ == "__main__":
    main()

import math
import io
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class HealDataset(Dataset):
    def __init__(self, root, size=256):
        self.files = (
            list(Path(root).rglob("*.jpg"))
            + list(Path(root).rglob("*.jpeg"))
            + list(Path(root).rglob("*.JPEG"))
            + list(Path(root).rglob("*.png"))
        )

        self.tf = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size),
            transforms.ToTensor(),  # [3,H,W], 0..1
        ])

    def __len__(self):
        return len(self.files)

    def _bezier_curve(self, points, num_samples):
        # Bernstein polynomial evaluation for arbitrary-degree Bezier
        n = len(points) - 1
        ts = torch.linspace(0, 1, num_samples)
        result = torch.zeros(num_samples, 2)
        for i, p in enumerate(points):
            b = math.comb(n, i) * (ts ** i) * ((1 - ts) ** (n - i))
            result += b.unsqueeze(1) * p
        return result

    def random_stroke_mask(self, h, w):
        radius = torch.randint(3, max(h, w) // 8, (1,)).item()
        n_pts = torch.randint(1, 6, (1,)).item()

        pts = torch.stack([
            torch.randint(0, w, (n_pts,)).float(),
            torch.randint(0, h, (n_pts,)).float(),
        ], dim=1)  # (N, 2) as (x, y)

        curve = self._bezier_curve(pts, max(h, w)) if n_pts > 1 else pts  # (S, 2)

        rows = torch.arange(h).float().view(h, 1, 1)
        cols = torch.arange(w).float().view(1, w, 1)
        cx = curve[:, 0].view(1, 1, -1)
        cy = curve[:, 1].view(1, 1, -1)
        mask = (((rows - cy) ** 2 + (cols - cx) ** 2) <= radius ** 2).any(dim=2)

        return mask.float().unsqueeze(0)

    def _corrupt(self, img):
        r = random.random()
        if r < 0.4:
            # gaussian noise
            img = (img + torch.randn_like(img) * random.uniform(0.02, 0.08)).clamp(0, 1)
        elif r < 0.7:
            # blur a random patch to simulate softened previous heal
            _, h, w = img.shape
            ph = random.randint(h // 8, h // 3)
            pw = random.randint(w // 8, w // 3)
            py = random.randint(0, h - ph)
            px = random.randint(0, w - pw)
            patch = img[:, py:py+ph, px:px+pw].unsqueeze(0)
            k = random.choice([5, 9, 13])
            sigma = random.uniform(1.5, 4.0)
            blurred = transforms.functional.gaussian_blur(patch.squeeze(0), k, sigma)
            img = img.clone()
            img[:, py:py+ph, px:px+pw] = blurred
        elif r < 0.9:
            # jpeg block artifacts
            pil = transforms.functional.to_pil_image(img)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=random.randint(30, 65))
            buf.seek(0)
            img = transforms.functional.to_tensor(Image.open(buf))
        # else: no corruption (~10% of the time)
        return img

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        clean = self.tf(img)

        _, h, w = clean.shape
        mask = self.random_stroke_mask(h, w)

        context = self._corrupt(clean) if random.random() < 0.4 else clean
        damaged = context * (1 - mask)

        # input is RGB + mask
        x = torch.cat([damaged, mask], dim=0)

        return x, clean, mask

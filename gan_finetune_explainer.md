# Adversarial fine-tuning for Healbrush

## Why

The current model is trained with `healing_loss` (L1 on hole + 10% L1 on valid
region) plus a small (0.1×) VGG perceptual term. Pixel-wise losses like L1
push the model toward the *average* of all plausible fills for a given hole —
that's the textbook cause of the smeared/blurred look you're seeing. An
adversarial loss fixes this directly: a discriminator is trained to spot
exactly that kind of blur, and its gradient pushes the generator away from it.

## What's new

### `Discriminator`
A small conditional PatchGAN-style critic. "Conditional" matters here — it
doesn't just ask "is this a real photo?" (which any sharp texture could
satisfy), it's shown the surrounding context and the hole mask too, so it asks
"does this fill make sense *for this hole*?". That's the signal that actually
pushes the generator toward content that fits its surroundings rather than
generically sharp noise.

It outputs a grid of patch-level logits rather than one whole-image verdict —
this rewards/punishes local texture quality, which is precisely the
granularity at which "blurry" shows up.

Spectral normalization on every conv keeps the discriminator from running away
with the game early — a standard GAN stabiliser.

### Hinge loss
`hinge_d_loss` / `hinge_g_loss` — the hinge formulation is more stable than
plain BCE for this kind of adversarial setup (less prone to vanishing
gradients once the discriminator gets confident), and is what most modern
GAN-based image-to-image / inpainting models use.

### Fine-tuning, not training from scratch
The generator (`UNet`) is architecturally untouched — the discriminator is a
wholly separate network that only exists to produce a training signal. So we
**resume the existing checkpoint** for the generator and start the
discriminator fresh. This is the standard way to bolt adversarial training
onto an already-converged reconstruction model (à la ESRGAN): the generator
already produces plausible-but-blurry output, so the discriminator has to
learn the *specific* failure mode you care about rather than distinguishing
"obvious garbage" from real photos — which is exactly the gradient you want.

When the checkpoint has no `disc` entry yet (i.e. this is the first GAN run),
the generator's LR is dropped from `1e-4` to `FINE_TUNE_LR = 5e-5`. This
matters: you're nudging a converged model with a new, noisier loss term, and
keeping the original LR risks the adversarial gradient knocking it out of the
minimum it already found before the discriminator has learned anything useful.

### The training step
Per batch, in order:
1. Generator forward pass (`pred = model(x)`).
2. **Discriminator step** — judge `clean` vs. `pred.detach()` (detached so
   gradients don't flow back into the generator here), step `opt_d`.
3. **Generator step** — combine the existing reconstruction terms with a new
   adversarial term: `healing_loss + 0.1·perceptual_loss + ADV_WEIGHT·adv_loss`,
   step `opt`.

`ADV_WEIGHT = 0.05` is intentionally small relative to the existing terms —
GAN losses can dominate and destabilise training if weighted too heavily. Watch
`d_loss` and `adv_loss` in wandb: if `d_loss` collapses toward 0 quickly, the
discriminator is overpowering the generator (lower its LR or `ADV_WEIGHT`
further); if `adv_loss` swings wildly, the generator is fighting it too hard
(lower `ADV_WEIGHT`).

### Checkpointing
`disc`/`opt_d` state is now saved alongside `model`/`opt` in both the periodic
checkpoint and `final.pt`, and loaded back in on resume — so subsequent runs
continue the GAN fine-tune rather than restarting it.

## What to watch for

- **Stability, not correctness** — this is the part that takes iteration.
  GANs can oscillate; expect to tune `ADV_WEIGHT` and the discriminator LR a
  few times based on the loss curves and the eval grid.
- **The eval grid is now your best signal for "did this help"** — look for
  sharper, more textured fills that still respect the surrounding content (not
  just generically sharp noise, which an unconditional discriminator could be
  fooled by but a conditional one shouldn't reward).

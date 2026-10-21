# Models and provenance

## Availability

The **[models-v1 release](https://github.com/bongsang/CTL-GAN/releases/tag/models-v1)**
provides SMC and SYSU generator weights for conditional image synthesis.
Checkpoint provenance and inference validation are documented below.

`maker` automatically downloads the selected model on first use, caches it, and
verifies its size and SHA-256 before every load. It uses safetensors; the public
CLI never deserializes pickle. No clinical source photographs, discriminator,
optimizer, training dataset paths, or embedded checkpoint source code are included
in these tensor files.

| Model | Direct download |
|---|---|
| SMC | [ctlgan-smc-v1.safetensors](https://github.com/bongsang/CTL-GAN/releases/download/models-v1/ctlgan-smc-v1.safetensors) |
| SYSU | [ctlgan-sysu-v1.safetensors](https://github.com/bongsang/CTL-GAN/releases/download/models-v1/ctlgan-sysu-v1.safetensors) |

For manual or offline installation, download the required file into a directory:

```bash
uv run -m maker --hospital smc --grade 3 --num 1 --cache-dir /path/to/models --offline
```

The release also includes provenance JSON, `SHA256SUMS`, and the NVIDIA license.
Weights are hosted separately from Git; each release uses immutable filenames
and registered hashes. Do not rename a different model to match these filenames.

## Released artifacts

Both are six-class, 512×512 RGB StyleGAN3-T `G_ema` generators (latent dimension
512). They contain 116 named parameters/buffers and **100,578,252 bytes each**
(about 95.9 MiB). Architecture settings and full hashes are in
[`models/registry.json`](../models/registry.json).

| Model | Source checkpoint | SHA-256 of safetensors |
|---|---|---|
| `ctlgan-smc-v1` | SMC `network-snapshot-025000.pkl` | `10fce586c6a5dbf64738158d04f9a9a84d37329747027112488d7398bbc7fb06` |
| `ctlgan-sysu-v1` | SYSU run `00006`, `network-snapshot-000900.pkl` | `7175bf28df87a602f3e655c96eda3163c72266d29050a91ad669ad4161583ef6` |

The retained SYSU generation command explicitly selects run 00006 / snapshot
000900, classes 0–5, seeds 0–999, and truncation 1. Its training configuration
resumes from the SMC snapshot 025000. The released weights support the generation
interface. Their correspondence with every image and quantitative result in the
paper has not been established.
The numeric checkpoint suffixes are training image counts in thousands (`kimg`),
not epochs.

For these exports, the checkpoint-embedded generator source exactly matches the
vendored `training/networks_stylegan3.py`. Every tensor equals its corresponding
tensor after reconstruction with that source. Validation of generated pixels and
historical examples is recorded in [validation.md](validation.md).

## What a model choice means

`smc` and `sysu` select separate generator weights. Grade 0–5 is a one-hot class
condition. The CLI does not retrieve a real patient's photograph or establish
that each generated image satisfies its requested clinical grade. The paper
notes that individual generated grades were not independently validated by
clinical grading and that no differential privacy guarantee was established.

These models are intended for non-commercial research/evaluation under the
[StyleGAN3 license](../LICENSE.txt). The downloadable artifacts are generated
from StyleGAN3 model checkpoints; the article's Creative Commons license does
not supersede the software terms.

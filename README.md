# CTL-GAN · Generate synthetic dry eye images

[![Paper](https://img.shields.io/badge/Scientific_Reports-2026-0066cc)](https://doi.org/10.1038/s41598-026-76373-4)
[![Maker checks](https://github.com/bongsang/CTL-GAN/actions/workflows/ci.yml/badge.svg)](https://github.com/bongsang/CTL-GAN/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](.python-version)

Generate **512 × 512 synthetic fluorescein-stained anterior segment images**,
conditioned on an SMC or SYSU model and an Oxford grade from 0 to 5.
The released SMC and SYSU weights support reproducible generation from random seeds.

**Watch the generator in action · 8-second demo**

https://github.com/user-attachments/assets/b12eecfa-55cf-4b7d-bfc0-bede2c17b3bf

[Watch or download the original MP4](docs/media/generating.mp4)

Images are computed from random latent inputs and learned weights, without
retrieving or assembling training photographs. The clip shows generated outputs;
[the generation pathway](docs/media/README.md#how-stylegan3-creates-an-image)
explains how each image is synthesized.

*Copyright © Bongsang Kim. You may use and share this video with author credit and
[citation of the CTL-GAN paper](https://doi.org/10.1038/s41598-026-76373-4).
[Video details and permission](docs/media/README.md).*

```bash
uv run -m maker --hospital smc --grade 3 --count 100
```

**[Paper](https://www.nature.com/articles/s41598-026-76373-4)** ·
**[Model details](docs/models.md)** ·
**[Installation & troubleshooting](docs/installation.md)** · **[Citations](CITATION.bib)**

> **Ready to generate:** [SMC and SYSU model weights](https://github.com/bongsang/CTL-GAN/releases/tag/models-v1)
> download automatically on first use (about 96 MiB per model), then stay cached.
> No training data or bulk image download is required.

## Quick start

The tested target is **Linux x86_64 with an NVIDIA GPU**. CPU inference is also
available for a small trial. Windows/WSL2 and macOS notes are in the
[installation guide](docs/installation.md).

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
2. Clone this repository and install the locked environment:

   ```bash
   git clone https://github.com/bongsang/CTL-GAN.git
   cd CTL-GAN
   uv python install 3.10.21
   uv sync --locked
   uv run -m maker --doctor
   ```

3. Generate images (the selected model downloads automatically on first use):

   ```bash
   # 100 SMC-domain images, Oxford grade 3, seeds 0–99
   uv run -m maker --hospital smc --grade 3 --count 100

   # 10 SYSU-domain images, grade 2, seeds 1000–1009
   uv run -m maker --hospital sysu --grade 2 --count 10 --seed 1000
   ```

The default `--device auto` uses CUDA when available. The default `--impl ref`
uses NVIDIA's reference operators on that device, so **no CUDA toolkit, nvcc, or
C++ compiler is needed for the default inference path**. The first environment
installation downloads PyTorch and its runtime libraries and can take several minutes.

### Output

Each command creates a separate directory under `outputs/<hospital>/grade-<n>/`:

```text
outputs/smc/grade-3/<run-id>/
├── smc_grade-3_seed-000000.png
├── smc_grade-3_seed-000001.png
├── …
└── manifest.json
```

Every PNG carries a synthetic-image description. The manifest records the
checkpoint hash, requested grade, seed, truncation, runtime, image hashes, and
completion status. Existing output directories are never overwritten.

## Generation options

| Option | Meaning | Default |
|---|---|---|
| `--hospital` | SMC or SYSU generator weights | Required |
| `--grade` | Requested Oxford condition, 0–5 | Required |
| `--count` | Number of images, 1–10,000 | `1` |
| `--seed` | Starting seed; each image uses the next seed | `0` |
| `--trunc` | Style distance from the learned mean, 0–2 | `1.0` |

### Examples

```bash
# See model availability and all options
uv run -m maker --list-models
uv run -m maker --help

# One image on CPU; useful for trying the interface without an NVIDIA GPU
uv run -m maker --hospital smc --grade 0 --count 1 --device cpu

# A chosen new output directory and a different seed range
uv run -m maker --hospital sysu --grade 4 --count 20 --seed 200 --outdir outputs/my-run

# Explore truncation (changes the sampling distribution)
uv run -m maker --hospital smc --grade 3 --count 4 --trunc 0.7

# Download and verify a model without generating images
uv run -m maker --hospital smc --download-only

# Use a pre-populated cache with networking disabled
uv run -m maker --hospital smc --grade 3 --count 1 --offline --cache-dir /path/to/models
```

`--seed N --count K` uses independent seeds `N` through `N+K-1`. Reusing the same
model, seed, grade, truncation, device and software environment gives repeatable
samples. Different operator implementations or GPU/software versions can change
pixel values; cross-platform bit-for-bit equality is not promised. Defaults use
float32 and truncation 1.0.

### Explore truncation

`--trunc` controls how far the generated style lies from the model's learned
average. `1.0` leaves it unchanged; `0.7` moves it 30% toward the average. Lower
values reduce variation. The StyleGAN paper describes a potential image-quality
benefit at the expense of diversity; this is not a validated improvement in
clinical quality for CTL-GAN. See [StyleGAN, Appendix B](https://arxiv.org/html/1812.04948v3#A2).

Compare the same seed and grade with two settings:

```bash
uv run -m maker --hospital smc --grade 3 --count 1 --seed 42 --trunc 1.0
uv run -m maker --hospital smc --grade 3 --count 1 --seed 42 --trunc 0.7
```

The average is shared across grades. Very low values can weaken grade-dependent
variation; at `0`, all styles collapse to that average, eliminating seed and grade
variation within a model. Start with `1.0`; `0.7` is an exploratory comparison.
[Truncation details and references](docs/sampling.md) explain the calculation and
values above `1`.

For the upstream compiled CUDA operators, use `--impl cuda` only after installing
a CUDA toolkit that matches the PyTorch CUDA runtime and a compatible compiler.
The reference path is the supported starting point; see [compatibility details](docs/installation.md).

## Example images

The repository includes 1,200 synthetic examples (100 per grade per model) in
[`dataset/smc-gen-ai-images`](dataset/smc-gen-ai-images) and
[`dataset/sysu-gen-ai-images`](dataset/sysu-gen-ai-images). No additional bulk image
dataset is required for generation. Model weights are cached outside Git.

| Existing SMC example · requested grade 3 | Existing SYSU example · requested grade 3 |
|---|---|
| ![Previously published synthetic SMC grade-3 example](dataset/smc-gen-ai-images/3/seed0000.png) | ![Previously published synthetic SYSU grade-3 example](dataset/sysu-gen-ai-images/3/seed0000.png) |

Numerical comparisons between these examples and current generator outputs are
reported in the [validation record](docs/validation.md).

## Overview

CTL-GAN transfers a StyleGAN3 generator trained on Samsung Medical Center (SMC)
images and adapts it using the Sun Yat-sen University (SYSU) dataset. This repository
provides a small inference interface around **[NVIDIA's StyleGAN3](https://github.com/NVlabs/stylegan3)**.
CTL-GAN applies collaborative transfer learning to the StyleGAN3 architecture.
[Method overview and diagrams](docs/method.md) describe the relationship between
the research workflow and this generation interface.

| Model option | Meaning | Output |
|---|---|---|
| `--hospital smc` | Generator trained on the SMC dataset | Synthetic SMC-domain images |
| `--hospital sysu` | Generator adapted from SMC weights using SYSU data | Synthetic SYSU-domain images |
| `--grade 0` … `--grade 5` | Requested Oxford grading condition | One-hot class conditioning |

These are **generated images**, not photographs retrieved from a hospital. The
requested grade is a conditioning label, not an independently verified clinical
grade. SYSU refers to the research dataset associated with Sun Yat-sen University,
not a live connection to a hospital.

## Research scope

This is an **image-generation companion** to the paper. It does not rerun the
classifier experiments or reproduce the paper's quantitative tables. The
publication discusses limitations including possible information leakage from
shared model weights and the lack of formal privacy guarantees. Image synthesis
does not establish anonymization, differential privacy, or suitability for clinical
decisions. Use the outputs for research and evaluation, not diagnosis.

The original clinical photographs are not needed by the generator and are not
included in model downloads. See the paper for dataset access conditions.

## Cite the work

Bongsang Kim, Yeo Kyoung Won, Hokyun Kim, Jiho Cha, Dong Hui Lim, and Jooyoung Jeon.
**Privacy-preserving collaborative transfer learning with generative adversarial
networks (CTL-GAN) for classification of dry eye disease.** *Nature Scientific Reports*
(2026). [doi:10.1038/s41598-026-76373-4](https://doi.org/10.1038/s41598-026-76373-4).
Published 20 October 2026.

Please also cite **Karras et al., Alias-Free Generative Adversarial Networks,
NeurIPS 2021 (StyleGAN3)** when using the generator. The paper entries, together
with the original StyleGAN truncation reference, are provided in
[`CITATION.bib`](CITATION.bib), and GitHub's “Cite this repository” uses
[`CITATION.cff`](CITATION.cff).

## Development and license

```bash
uv run -m unittest discover -s tests -v
```

The small CLI tests run without model downloads or a GPU. Actual checkpoint
validation is recorded separately in [docs/validation.md](docs/validation.md).
Training scripts and upstream documentation are retained for reference; the
locked maker environment is validated for inference, not for reproducing training.
Use the [upstream repository](https://github.com/NVlabs/stylegan3) for training tools.

The **NVIDIA Source Code License for StyleGAN3** applies; its use is limited to
non-commercial research/evaluation. Read [`LICENSE.txt`](LICENSE.txt) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The article's publication
license does not replace the software license.

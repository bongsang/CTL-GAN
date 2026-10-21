# Seeds and truncation

`--count K --seed N` generates K images with seeds N through N+K−1. Each seed
initializes a separate pseudorandom draw of a 512-dimensional latent vector.
Repeating a model, grade, seed, truncation setting and runtime reproduces the
output. Changing the seed samples a new latent input; consecutive seeds do not
represent adjacent points in latent space.

## Truncation

StyleGAN's truncation trick scales the style vector's displacement from its mean:

```text
w_adjusted = w_average + psi × (w − w_average)
```

Here, `psi` is `--trunc`. The average is a latent representation, not an averaged
photograph. Appendix B of [Karras et al., A Style-Based Generator Architecture for
Generative Adversarial Networks (CVPR 2019)](https://arxiv.org/html/1812.04948v3#A2)
describes this quality–diversity trade-off. The released StyleGAN3 implementation
applies the operation in its [mapping network](../training/networks_stylegan3.py).

| Value | Effect in this implementation |
|---|---|
| `1.0` | Default: use the mapped style unchanged |
| `0.7` | Retain 70% of its displacement from the learned mean |
| `0 < psi < 1` | Move styles toward the mean, reducing their spread |
| `0` | Use the shared mean for all styles; seed and grade variation disappear within the selected model |
| `1 < psi <= 2` | Extrapolate away from the mean; artifacts may increase |

The maker applies truncation to all style inputs. Each checkpoint stores one
`w_avg` shared across grades, rather than a separate mean for each grade. Strong
truncation therefore attenuates conditioning differences as well as differences
between seeds. This follows from the implementation; no clinical improvement or
optimal truncation value has been established for these released models.

## Compare settings

Keep the hospital, grade and seed fixed to isolate truncation:

```bash
uv run -m maker --hospital smc --grade 3 --count 1 --seed 42 --trunc 1.0
uv run -m maker --hospital smc --grade 3 --count 1 --seed 42 --trunc 0.7
```

Each command creates its own output directory. The manifest records
`truncation_psi` and each image's seed. Changing truncation transforms the same
mapped sample; it does not draw a new independent sample. For new samples, change
`--seed` or increase `--count`.

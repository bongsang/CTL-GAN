# Models and provenance

## Availability

Two generator-only exports have been prepared locally. **They are not yet public
release assets.** The registry deliberately has `url: null` until the author
confirms the selected checkpoints and their public release. Installation alone
therefore does not enable first-time generation yet.

For a prepared local export, put both `.safetensors` files in a cache directory:

```bash
uv run -m maker --hospital smc --grade 3 --num 1 --cache-dir /path/to/models --offline
```

Once published, release assets will be hosted separately from Git and downloaded
on demand. `maker` verifies the file size and SHA-256 before loading, including
on cached runs. It uses safetensors; the public CLI never deserializes pickle.
No clinical source photographs, discriminator, optimizer, training dataset paths,
or embedded checkpoint source code are included in these tensor files.

## Prepared artifacts

Both are six-class, 512×512 RGB StyleGAN3-T `G_ema` generators (latent dimension
512). They contain 116 named parameters/buffers and **100,578,252 bytes each**
(about 95.9 MiB). Architecture settings and full hashes are in
[`models/registry.json`](../models/registry.json).

| Model | Historical checkpoint candidate | SHA-256 of safetensors |
|---|---|---|
| `ctlgan-smc-v1` | SMC `network-snapshot-025000.pkl` | `10fce586c6a5dbf64738158d04f9a9a84d37329747027112488d7398bbc7fb06` |
| `ctlgan-sysu-v1` | SYSU run `00006`, `network-snapshot-000900.pkl` | `7175bf28df87a602f3e655c96eda3163c72266d29050a91ad669ad4161583ef6` |

The retained SYSU generation command explicitly selects run 00006 / snapshot
000900, classes 0–5, seeds 0–999, and truncation 1. Its training configuration
resumes from the SMC snapshot 025000. These are evidence-backed **candidate demo
models**; that evidence alone does not establish that every final paper image or
reported metric came from these exact checkpoint bytes. Author confirmation and
verification against the complete historical image pool are separate matters.
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

## Maintainer export procedure

Use only a checkpoint that the maintainer already trusts: pickle loading can
execute code. The public inference command never needs this step.

```bash
uv run scripts/export_generator.py \
  --hospital smc \
  --checkpoint /absolute/path/to/network-snapshot-025000.pkl \
  --output-dir artifacts/models \
  --trust-pickle
```

The exporter reads the original without modifying it, rejects a generator with
a different embedded implementation, reconstructs the generator, checks every
tensor, and writes safetensors plus a provenance JSON. Before publishing a new
version, validate actual generation, assign a new model ID/filename, register
its hash and architecture, and publish immutable release assets. Never silently
replace an existing model version. The command refuses to overwrite exports.

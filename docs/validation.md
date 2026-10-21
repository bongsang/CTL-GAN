# Validation record

Checked on **2026-10-21**. These are inference compatibility checks, not a
reproduction of the paper's classifier results or a clinical quality evaluation.

## Environment

- Linux x86_64, kernel 6.12.113, glibc 2.41.
- Python 3.10.21, uv 0.13.0.
- PyTorch 2.1.2+cu121; NumPy 1.26.4; SciPy 1.11.4; Pillow 10.4.0.
- NVIDIA GeForce RTX 4090; NVIDIA driver 590.48.01.
- No system CUDA compiler (`nvcc` unavailable).
- NVIDIA reference operators (`--impl ref`), float32, truncation 1.0.
- cuDNN deterministic mode on; TF32 and cuDNN benchmarking off.

## Actual model execution

| Check | Result |
|---|---|
| Reconstruct SMC and SYSU generators from safetensors | Strict state loading passed; 116 tensors per model preserved exactly |
| Embedded generator source vs vendored source | Exact source match for both checkpoints |
| Original pickle vs exported model, seed 0, both models × all 6 grades | **12/12 pixel arrays exactly equal** in the same reference inference runtime |
| SMC, grade 3, seeds 0–99 | **100 valid 512×512 RGB PNGs**, 100 distinct hashes, complete manifest |
| Repeat SMC grade 3 / seed 0 in a fresh CLI process | PNG bytes exactly equal |
| SYSU, grade 3, seeds 0–1 | Two valid PNGs and complete manifest |
| CPU: SYSU, grade 2, seed 0 | One valid PNG and complete manifest |
| Ordinary CLI, download and manifest tests | 13 passed |
| Public Release download from an empty cache | Both models downloaded without GitHub authentication and passed size/SHA-256 verification |
| Inference with publicly downloaded weights | SMC: 3 PNGs; SYSU: 1 PNG; complete manifests; SMC seed-0 PNG equals the local-export PNG |
| GitHub Actions | Locked dependency resolution, 13 tests, CLI help/model list and CFF schema validation passed |

The 100-image SMC loop took **18.316 seconds** on this RTX 4090, including PNG
and manifest writes, **excluding environment installation, model download,
verification and initialization**. The single SYSU CPU generation loop took
6.177 seconds with eight PyTorch threads. These observations are not a cross-device
performance guarantee or a formal benchmark.

The two original training pickles were used only for local maintainer validation.
They are not part of the public inference path or Git changes.

## Existing public examples

All twelve existing `dataset/<hospital>-gen-ai-images/<grade>/seed0000.png`
examples were compared to fresh output. Mean absolute pixel differences on the
0–255 scale were:

| Oxford condition | SMC MAE | SYSU MAE |
|---|---:|---:|
| 0 | 0.0462 | 0.0333 |
| 1 | 0.0534 | 0.0455 |
| 2 | 0.0648 | 0.0342 |
| 3 | 0.0610 | 0.0304 |
| 4 | 0.0716 | 0.0352 |
| 5 | 0.0606 | 0.0400 |

Maximum single-channel differences were 3/255 for SMC and 2/255 for SYSU.
This is strong numerical correspondence with these selected public examples,
but **not byte-identical historical reproduction**. The historical runtime and
precision for those examples are not proven by this comparison. Do not attribute
all remaining differences to one cause without further testing.

The full per-grade comparison is available in
[`generator-validation.json`](generator-validation.json). This does not establish
that every image used in the final paper came from these checkpoint bytes.

## Re-run the interface checks

```bash
uv sync --locked
uv run -m unittest discover -s tests -v
uv run -m maker --doctor
uv run -m maker --hospital smc --grade 3 --num 100 --outdir outputs/validation-smc
uv run -m maker --hospital sysu --grade 2 --num 1 --device cpu --outdir outputs/validation-cpu
```

Model files must first be available as described in [models.md](models.md).
Choose new output directories on subsequent runs. CI runs the standard-library
CLI/control-plane tests; it does not download hospital models or claim GPU tests.

Not validated: native Windows, macOS, MPS, compiled CUDA extensions, other GPU
architectures, training in this environment, clinical label correctness, formal
privacy guarantees, or all historical image-pool/checkpoint relationships.

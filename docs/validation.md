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
| Ordinary CLI, download and manifest tests | 14 passed, including count and truncation argument checks |
| Public Release download from an empty cache | Both models downloaded without GitHub authentication and passed size/SHA-256 verification |
| Inference with publicly downloaded weights | SMC: 3 PNGs; SYSU: 1 PNG; complete manifests; SMC seed-0 PNG equals the local-export PNG |
| GitHub Actions | Locked dependency resolution, 14 tests, CLI help/model list and CFF schema validation passed |

The 100-image SMC loop took **18.316 seconds** on this RTX 4090, including PNG
and manifest writes, **excluding environment installation, model download,
verification and initialization**. The single SYSU CPU generation loop took
6.177 seconds with eight PyTorch threads. These observations are not a cross-device
performance guarantee or a formal benchmark.

The two original training pickles were used only for local maintainer validation.
The public inference path uses the exported safetensors files.

## Count and truncation checks

The `--count` interface was checked with the released SMC model on the environment
above. Seven images were generated: seeds 42–43 at grade 3 with truncation 1.0,
0.7 and 0.0, plus seed 100 at grade 2 with truncation 0.0. All runs produced the
requested number of 512×512 RGB images and complete manifests with the expected
seeds and truncation values.

Seeds 42 and 43 produced different pixel arrays at truncation 1.0. Holding seed 42
and grade 3 fixed, truncation 0.7 changed the output. At truncation 0.0, all three
outputs were pixel-identical across the tested seeds and grades, consistent with
the shared style mean. These checks verify sampling behavior, not image quality.
The removed `--num` option is rejected by the argument parser.

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
The selected examples show close numerical agreement, although the outputs are
not byte-identical. The historical runtime and numerical precision for these
examples are undocumented, and the source of the remaining differences has not
been isolated.

The full per-grade comparison is available in
[`generator-validation.json`](generator-validation.json). This does not establish
that every image used in the final paper came from these checkpoint bytes.

## Re-run the interface checks

```bash
uv sync --locked
uv run -m unittest discover -s tests -v
uv run -m maker --doctor
uv run -m maker --hospital smc --grade 3 --count 100 --outdir outputs/validation-smc
uv run -m maker --hospital sysu --grade 2 --count 1 --device cpu --outdir outputs/validation-cpu
```

Model files must first be available as described in [models.md](models.md).
Choose new output directories on subsequent runs. CI runs the standard-library
CLI and model-download logic tests without downloading hospital models or
running GPU inference.

Not validated: native Windows, macOS, MPS, compiled CUDA extensions, other GPU
architectures, training in this environment, clinical label correctness, formal
privacy guarantees, or all historical image-pool/checkpoint relationships.

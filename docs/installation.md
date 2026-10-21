# Installation and compatibility

## Why Python is pinned

StyleGAN3 was developed with Python 3.8 and PyTorch 1.9-era dependencies. Installing
its original `environment.yml` with today's newest Python/PyTorch/NumPy is not a
reliable inference setup. In particular, the old operators and serialized models
have assumptions that newer libraries can break.

The inference environment uses the following pinned dependencies:

| Component | Pinned version | Reason |
|---|---|---|
| Python | 3.10.21 (`>=3.10,<3.11`) | Explicit interpreter selection and compatible binary wheels |
| PyTorch | 2.1.2 | Compatibility profile for the existing StyleGAN3 implementation |
| NumPy | 1.26.4 | Stay below NumPy 2 for this PyTorch binary |
| SciPy | 1.11.4 | Compatible signal-processing routines and Python wheels |
| Pillow | 10.4.0 | PNG output |
| setuptools | 75.8.2 | Retains `pkg_resources`, imported by upstream operators |
| safetensors | 0.4.5 | Non-executable tensor storage for public models |

`uv.lock` pins transitive dependencies and artifact hashes for the validated
inference environment. Install with `uv sync --locked` to retain these versions.
The upstream `environment.yml` records the original StyleGAN3 configuration;
the maker environment is validated for inference and does not reconstruct the
historical CTL-GAN training environment.

## Linux / NVIDIA GPU

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone --depth 1 https://github.com/bongsang/CTL-GAN.git
cd CTL-GAN
uv python install 3.10.21
uv sync --locked
uv run -m maker --doctor
```

On Linux x86_64, the pinned PyTorch wheel uses CUDA 12.1 runtime libraries. Install
a compatible NVIDIA driver on the host. A system CUDA toolkit is unnecessary
for default `--impl ref`. `nvidia-smi`'s displayed CUDA version is the driver's
supported level; it need not equal the PyTorch runtime version.

The installer requires several GB for PyTorch and GPU libraries. Models are
fetched separately and verified by SHA-256. The process generates one image at
a time, keeping memory requirements independent of `--count`. See the measured
hardware in [validation.md](validation.md). Lower-memory GPUs have not been tested.

## CPU, Windows, and macOS

- **CPU:** `--device cpu --count 1` uses the same reference operators. It is slower;
  start with one image. On Linux, the default environment still installs the
  GPU-capable PyTorch wheel even when generating on CPU.
- **Windows:** WSL2 with an NVIDIA-compatible host driver is the recommended
  starting point. Native Windows is not validated by this release.
- **macOS:** CPU inference may be possible with compatible PyTorch 2.1.2 wheels,
  but is not validated here. MPS is not an implemented device option.
- **New GPU generations:** the pinned PyTorch/CUDA version may lack support for
  newer architectures (e.g. Blackwell). Use CPU or a separately validated runtime;
  installing the latest Python into this environment is not a compatibility fix.

## Optional compiled operators

`--impl cuda` selects NVIDIA's optimized extension path. It needs an NVIDIA GPU,
a full CUDA toolkit with `nvcc` matching PyTorch's CUDA runtime, and a compatible
host compiler. The PyTorch wheel includes CUDA runtime libraries but **does not
include that compiler**. Extension builds occur on first use and are cached by
PyTorch. Set `TORCH_EXTENSIONS_DIR` to a writable location if needed.

The default reference implementation bypasses extension compilation. It uses
PyTorch operations on GPU or CPU and keeps the same learned model weights, but
may differ numerically from the fused kernels. The compiled path has not been
validated for this release.

## Common problems

| Symptom | Action |
|---|---|
| Wrong Python / unsupported wheel | Run from the repository root; `uv python install 3.10.21`, then `uv sync --locked` |
| Model download fails | Check connectivity to GitHub Releases; use the manual download in [models.md](models.md) |
| CUDA unavailable | Check `nvidia-smi` and `uv run -m maker --doctor`; try `--device cpu` |
| `nvcc` / extension build failure | Use the default `--impl ref`, or install the matching toolkit and compiler |
| `pkg_resources` missing | Restore the locked environment with `uv sync --locked` |
| NumPy ABI or `_ARRAY_API` error | Restore NumPy 1.26.4 via `uv sync --locked` |
| SHA-256 verification failure | Move the damaged cache file aside, then redownload; never bypass the check |
| Output directory already exists | Choose another `--outdir`; existing results are preserved |
| Interrupted download | Retry; an incomplete temporary file is not promoted to the model cache |
| Interrupted generation | Completed PNGs and `manifest.json` remain; use a new output directory for another run |

## Offline use

On a connected machine, download each required model:

```bash
uv run -m maker --hospital smc --download-only --cache-dir ./models-cache
uv run -m maker --hospital sysu --download-only --cache-dir ./models-cache
```

Copy that directory to the offline machine after preparing its Python environment:

```bash
uv run -m maker --hospital smc --grade 3 --count 1 --offline --cache-dir ./models-cache
```

Model cache precedence is `--cache-dir`, then `CTL_GAN_CACHE`, then
`$XDG_CACHE_HOME/ctl-gan` (or `~/.cache/ctl-gan`). `--offline` controls model
networking; prepare the uv environment before disconnecting. `uv run --offline`
can also disable uv's own package networking.

"""Run with: uv run -m maker --hospital smc --grade 3 --num 100."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from . import __version__
from .models import MakerError, cache_root, ensure_model, registry, sha256


def positive(value):
    number = int(value)
    if not 1 <= number <= 10000:
        raise argparse.ArgumentTypeError("must be between 1 and 10000")
    return number


def seed_value(value):
    number = int(value)
    if not 0 <= number <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return number


def trunc_value(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 2:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 2")
    return number


def parser():
    p = argparse.ArgumentParser(description="Generate synthetic dry eye images conditioned on hospital model and Oxford grade (0–5).")
    p.add_argument("--hospital", choices=("smc", "sysu"), help="SMC source model or SYSU-adapted model")
    p.add_argument("--grade", type=int, choices=range(6), help="requested Oxford condition, not a verified clinical label")
    p.add_argument("--num", type=positive, default=1, help="number of images, 1–10000 (default: 1)")
    p.add_argument("--seed", type=seed_value, default=0, help="first seed; generates seed through seed+num-1")
    p.add_argument("--trunc", type=trunc_value, default=1.0, help="truncation psi (default: 1.0)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--impl", choices=("ref", "cuda"), default="ref", help="ref needs no compiler; cuda needs a matching CUDA toolkit")
    p.add_argument("--outdir", type=Path, help="new output directory (existing directories are never overwritten)")
    p.add_argument("--cache-dir", type=Path, default=cache_root())
    p.add_argument("--offline", action="store_true", help="use only an already cached, verified model")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--doctor", action="store_true", help="show installed runtime and GPU diagnostics")
    p.add_argument("--version", action="version", version=__version__)
    return p


def write_manifest(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args, p):
    catalog = registry()
    if args.list_models:
        for name, spec in catalog["models"].items():
            print(f"{name}: {spec['label']} | Oxford 0–5 | 512×512 | {spec['status']}")
        return
    if args.doctor:
        from .runtime import environment
        print(json.dumps(environment(), indent=2))
        return
    if not args.hospital:
        p.error("--hospital is required (or use --list-models / --doctor)")
    if not args.download_only and args.grade is None:
        p.error("--grade is required for generation")
    if args.seed + args.num - 1 > 2**32 - 1:
        p.error("seed + num - 1 must not exceed 4294967295")
    if args.outdir is not None and args.outdir.exists() and not args.download_only:
        raise MakerError(f"Output directory already exists: {args.outdir}. Choose a new --outdir.")
    spec = catalog["models"][args.hospital]
    print(f"Checking {args.hospital} model …", flush=True)
    path = ensure_model(spec, args.cache_dir.expanduser(), args.offline)
    if args.download_only:
        print(f"Verified model: {path}")
        return
    from .runtime import environment, load_generator, operators, select_device, synthesize
    from PIL import Image, PngImagePlugin
    device = select_device(args.device)
    if args.impl == "cuda":
        from torch.utils.cpp_extension import CUDA_HOME
        if device != "cuda" or CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
            raise MakerError("--impl cuda requires a CUDA GPU and CUDA toolkit with nvcc. Use --impl ref (default) without a compiler.")
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = load_generator(path, spec, device)
    out = args.outdir or Path("outputs") / args.hospital / f"grade-{args.grade}" / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "status": "running", "synthetic": True,
        "created_at": datetime.now(timezone.utc).isoformat(), "maker_version": __version__,
        "paper_doi": "10.1038/s41598-026-76373-4", "hospital": args.hospital,
        "requested_oxford_grade": args.grade, "clinically_verified_grade": False,
        "model_id": spec["id"], "model_sha256": spec["sha256"],
        "upstream_commit": catalog["stylegan3_commit"],
        "seed_start": args.seed, "num_requested": args.num, "truncation_psi": args.trunc,
        "device": device, "operator_implementation": args.impl, "precision": "float32",
        "environment": environment(), "images": [],
    }
    target = out / "manifest.json"
    write_manifest(target, manifest)
    started = time.monotonic()
    try:
        with operators(args.impl):
            for index, seed in enumerate(range(args.seed, args.seed + args.num), 1):
                pixels = synthesize(generator, args.grade, seed, args.trunc, device)
                filename = f"{args.hospital}_grade-{args.grade}_seed-{seed:06d}.png"
                info = PngImagePlugin.PngInfo()
                info.add_text("Description", f"CTL-GAN synthetic image; {args.hospital}; requested Oxford grade {args.grade}; seed {seed}. Not a patient photograph or verified clinical label.")
                Image.fromarray(pixels, "RGB").save(out / filename, pnginfo=info)
                manifest["images"].append({"filename": filename, "seed": seed, "sha256": sha256(out / filename)})
                write_manifest(target, manifest)
                print(f"[{index}/{args.num}] {filename}", flush=True)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_manifest(target, manifest)
    print(f"Saved {len(manifest['images'])} synthetic images and manifest.json to {out}")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        run(args, p)
    except (MakerError, OSError, RuntimeError, ValueError) as exc:
        print(f"maker: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("maker: interrupted; completed images and their manifest were retained.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

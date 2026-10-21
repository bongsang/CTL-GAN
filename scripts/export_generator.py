"""Maintainer-only conversion of an explicitly trusted local training checkpoint.

This is the only CTL-GAN utility that deserializes pickle. Never use an untrusted
checkpoint. The reader-facing maker CLI loads only verified safetensors.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maker.models import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hospital", choices=("smc", "sysu"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trust-pickle", action="store_true", required=True, help="acknowledge that this local checkpoint can execute code")
    args = p.parse_args()
    import torch
    from safetensors.torch import save_file
    from training.networks_stylegan3 import Generator
    with args.checkpoint.open("rb") as source:
        checkpoint = pickle.load(source)
    generator = checkpoint["G_ema"].eval().requires_grad_(False).cpu()
    source_code = (Path(__file__).resolve().parents[1] / "training/networks_stylegan3.py").read_text()
    if generator._orig_module_src.strip() != source_code.strip():
        raise RuntimeError("Checkpoint embeds a different generator implementation. Review it before export.")
    if generator.init_args:
        raise RuntimeError("Unexpected positional generator constructor arguments.")
    kwargs = dict(generator.init_kwargs)
    restored = Generator(**kwargs).eval().requires_grad_(False)
    state = {key: value.detach().cpu().contiguous().clone() for key, value in generator.state_dict().items()}
    restored.load_state_dict(state, strict=True)
    for key, tensor in restored.state_dict().items():
        if not torch.equal(tensor, state[key]):
            raise RuntimeError(f"Tensor changed during reconstruction: {key}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"ctlgan-{args.hospital}-v1.safetensors"
    if output.exists():
        raise FileExistsError(output)
    save_file(state, str(output), metadata={"format": "CTL-GAN G_ema", "hospital": args.hospital})
    metadata = {
        "id": f"ctlgan-{args.hospital}-v1", "filename": output.name,
        "sha256": sha256(output), "bytes": output.stat().st_size,
        "generator_kwargs": kwargs, "source_checkpoint_sha256": sha256(args.checkpoint),
        "tensor_count": len(state), "source_code_matches_vendored_generator": True,
        "all_tensors_equal_after_reconstruction": True,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

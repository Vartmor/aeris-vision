"""Verify PyTorch CPU and CUDA forward/backward execution."""

import argparse
import json
import platform
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("all", "cpu", "cuda"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    import torchvision

    report = {
        "python": platform.python_version(),
        "kernel": platform.release(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda_build": torch.version.cuda,
        "devices": {},
    }
    devices = ("cpu", "cuda") if args.device == "all" else (args.device,)
    for device in devices:
        try:
            if device == "cuda":
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats()
            values = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
            loss = values.square().sum()
            loss.backward()
            torch.testing.assert_close(loss.detach().cpu(), torch.tensor(14.0))
            torch.testing.assert_close(values.grad.cpu(), torch.tensor([2.0, 4.0, 6.0]))
        except (AssertionError, RuntimeError) as error:
            report["devices"][device] = {"passed": False, "error": str(error)}
            continue

        result = {"passed": True, "value": loss.item(), "gradient": values.grad.tolist()}
        if device == "cuda":
            torch.cuda.synchronize()
            result.update(
                name=torch.cuda.get_device_name(),
                total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
        report["devices"][device] = result

    output = json.dumps(report, indent=2)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    return 0 if all(result["passed"] for result in report["devices"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

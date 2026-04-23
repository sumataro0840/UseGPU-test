import time

import torch


class GPUBenchmark:
    def __init__(self, matrix_size: int = 10000) -> None:
        self.matrix_size = matrix_size

    def _create_matrices(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randn(self.matrix_size, self.matrix_size, device=device),
            torch.randn(self.matrix_size, self.matrix_size, device=device),
        )

    def run_cpu(self) -> float:
        a, b = self._create_matrices("cpu")
        start = time.time()
        _ = a @ b
        return time.time() - start

    def run_gpu(self) -> float | None:
        if not torch.cuda.is_available():
            return None

        a, b = self._create_matrices("cuda")
        torch.cuda.synchronize()
        start = time.time()
        _ = a @ b
        torch.cuda.synchronize()
        return time.time() - start

    def run(self) -> dict[str, float | None]:
        return {
            "cpu": self.run_cpu(),
            "gpu": self.run_gpu(),
        }

    def display(self) -> dict[str, float | None]:
        result = self.run()
        print(f"CPU: {result['cpu']:.4f}s")
        if result["gpu"] is None:
            print("GPU: CUDA is not available")
        else:
            print(f"GPU: {result['gpu']:.4f}s")
        return result

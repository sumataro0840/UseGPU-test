from .GPUvsCPU import GPUBenchmark
from .NoiseData import NoiseData

benchmark = GPUBenchmark()
noise_data = NoiseData()


def run_benchmark() -> dict[str, float | None]:
    return benchmark.display()


def generate_noise_data():
    return noise_data.display()


__all__ = [
    "GPUBenchmark",
    "NoiseData",
    "benchmark",
    "noise_data",
    "run_benchmark",
    "generate_noise_data",
]

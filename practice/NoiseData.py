import torch


class NoiseData:
    def __init__(
        self,
        start: float = 0.0,
        end: float = 10.0,
        steps: int = 1000,
        noise_level: float = 0.3,
    ) -> None:
        self.start = start
        self.end = end
        self.steps = steps
        self.noise_level = noise_level

    def generate(self) -> dict[str, torch.Tensor]:
        x = torch.linspace(self.start, self.end, self.steps)
        clean = torch.sin(x)
        noisy = clean + self.noise_level * torch.randn_like(clean)

        return {
            "x": x,
            "clean": clean,
            "noisy": noisy,
        }

    def display(self) -> dict[str, torch.Tensor]:
        data = self.generate()
        print("Noise data sample:")
        print(f"x[:5] = {data['x'][:5]}")
        print(f"clean[:5] = {data['clean'][:5]}")
        print(f"noisy[:5] = {data['noisy'][:5]}")
        return data

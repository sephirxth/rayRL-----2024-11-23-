from dataclasses import dataclass

@dataclass
class TrainConfig:
    """训练配置"""
    algorithm: str = "PPO"
    batch_size: int = 4096
    learning_rate: float = 1e-3
    gamma: float = 0.99
    num_workers: int = 4
    save_interval: int = 100
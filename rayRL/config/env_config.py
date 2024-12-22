from dataclasses import dataclass


@dataclass
class EnvConfig:
    """环境配置"""

    max_episodes: int = 1000
    max_sim_time: int = 8000
    render_mode: str = "rgb_array"
    yellow_time: int = 3
    min_green_time: int = 10
    max_green_time: int = 60
    render_modes = "human"  # "rgb_array"],
    render_fps = 4
    flash_episode: bool = (False,)
    sumocfg: str = None

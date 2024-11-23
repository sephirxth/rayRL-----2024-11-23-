"""
测试代码
"""

import traci
import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQN  # 更新后的导入路径
from ray.rllib.env import PettingZooEnv
from ray.tune.logger import pretty_print
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sumo_env import SumoEnv


# 智能体构建及训练函数
def train(max_episode, config_file, task_id):
    # 配置训练参数
    env = SumoEnv(
        render_mode="rgb_array",
        max_episodes=max_episode,
        max_sim_time=5000,
        sumocfg=config_file,
    )

    # 开始训练
    for i in range(max_episode):
        state, info = env.reset()
        done = False
        while not done:
            traci.simulationStep()
            state = env._get_combined_state()
            # print("state is =====>>",state)
            done = env.check_terminated()
        print("回合========>>>>>>:", i)
    env.close()
    return f"Task {task_id}: Simulation completed"


@ray.remote(num_cpus=5)
def run_sumo_simulation_remote(max_episode, config_file, task_id):
    return train(max_episode, config_file, task_id)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 使用绝对路径指定配置文件
    config_file = os.path.join(project_root, "sumo_xml", "three_points.sumocfg")
    config_files = [config_file] * 5
    max_episode = 200
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": project_root,  # 使用项目根目录作为工作目录
            "py_modules": [
                os.path.join(project_root, "env")  # 显式包含 env 模块
            ],
        },
    )
    print("Ray initialized:", ray.is_initialized())
    print("Current active workers:", ray.available_resources())

    # 启动并行任务，分配任务编号
    futures = [
        run_sumo_simulation_remote.remote(max_episode, cfg, task_id=i)
        for i, cfg in enumerate(config_files)
    ]
    results = ray.get(futures)

    # 输出结果
    print("\nResults:")
    for result in results:
        print(result)

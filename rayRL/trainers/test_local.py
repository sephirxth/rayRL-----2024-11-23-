"""
测试代码,获取固定配时车辆输出代码
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
from config.config import SUMO_CONFIG  # 引用配置字典


# 智能体构建及训练函数
def test_local(max_episode, config_file):
    # 配置训练参数
    env = SumoEnv(config=config_file)

    # 开始训练
    for i in range(max_episode):
        state, info = env.reset()
        done = False
        while not done:
            traci.simulationStep()
            state = env._get_combined_state()
            # reward = env.reward()
            # print(f"rewaid is =====>>, {reward}")
            done = env.check_terminated()
        print("回合========>>>>>>:", i)
    env.close()



if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 使用绝对路径指定配置文件
    config = SUMO_CONFIG["sumo_env"]
    max_episode = 200
    test_local(max_episode, config)


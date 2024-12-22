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
def train(max_episode, config_file):
    # 配置训练参数
    env = SumoEnv(
        #render_mode="rgb_array",
        render_mode="rgb_array",
        max_episodes=max_episode,
        max_sim_time=8000,
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
        print("current episode 回合========>>>>>>:", i)
    env.close()



if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 使用绝对路径指定配置文件
    config_file = os.path.join(project_root, "one_way_xml", "one_way.sumocfg")
    max_episode = 3
    train(max_episode, config_file)


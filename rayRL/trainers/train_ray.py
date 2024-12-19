import traci
import sys
import os
import ray
from ray import tune
import argparse
import matplotlib.pyplot as plt
from ray.rllib.algorithms.ppo import PPOConfig  
from ray.rllib.policy import Policy
from ray.tune.registry import register_env
from ray.tune import Callback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.sumo_env import SumoEnv

# 注册环境
def create_env(env_config):
    return SumoEnv(**env_config)
register_env("sumo_env", create_env)

class PPOSimulation:
    def __init__(self, max_episode, config_file):
        self.max_episode = max_episode
        self.config_file = config_file
        # 配置 PPO 参数
        self.ppo_config = PPOConfig()
        self.ppo_config.env = "sumo_env"
        self.ppo_config.env_config = {
            "render_mode": "rgb_array",
            "max_episodes": max_episode,
            "max_sim_time": 8000,
            "sumocfg": config_file,
        }
        self.ppo_config.train_batch_size = 128  
        self.ppo_config.minibatch_size = 32  
        self.ppo_config.gamma = 0.99
        self.ppo_config.lr = 1e-3
        self.ppo_config.num_workers = 1  
        self.ppo_config.num_gpus = 1 

        # 创建模型文件路径trainers/checkpoints
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_dir = os.path.join(current_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.plot_dir = os.path.join(current_dir, "plot")  
        os.makedirs(self.plot_dir, exist_ok=True)

        # 初始化 PPO 算法
        self.ppo_trainer = self.ppo_config.build()

    def train(self):
        for episode in range(self.max_episode):
            result = self.ppo_trainer.train()
            # 保存模型
            if episode % 10 == 0:
                checkpoint = self.ppo_trainer.save(self.checkpoint_dir)
                print(f"Checkpoint saved at {checkpoint}")
            print("回合========>>>>>>:", episode)

    def test(self, checkpoint_path):
        self.ppo_trainer.restore(checkpoint_path)
        env = SumoEnv(render_mode="rgb_array", max_episodes=1, max_sim_time=8000, sumocfg=self.config_file)
        state = env.reset()
        done = False
        total_reward = 0
        rewards = []
        print(self.ppo_trainer.config.to_dict())


        while not done:
            action = self.ppo_trainer.compute_single_action(state) 
            print(f"action is: ====={action}")
            state, reward, done, _ = env.step(action)
            total_reward += reward
            rewards.append(total_reward)
        self.plot_rewards(rewards)
        return total_reward

    def plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward Curve')
        plt.savefig(os.path.join(self.plot_dir, "reward_curve.png"))


if __name__ == "__main__":
    # 使用绝对路径指定配置文件
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(project_root, "one_way_xml", "one_way.sumocfg")
    
    # 每个task的训练回合数
    max_episode = 11

    # 启动 Ray
    ray.shutdown()  
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": project_root,  
            "py_modules": [
                os.path.join(project_root, "env")  
            ],
        },
    )

    ppo_sim = PPOSimulation(max_episode, config_file)
    ppo_sim.train()
    # 获取最佳检查点路径
    best_checkpoint = ppo_sim.ppo_trainer.save()
    test_reward = ppo_sim.test(best_checkpoint)
    print(f"Test reward: {test_reward}")
import traci
import os
import sys
import pathlib
import torch
import numpy as np
import ray
import logging
import time
from ray import tune
import argparse
import matplotlib.pyplot as plt
from ray.rllib.core.rl_module import RLModule
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
import configparser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.sumo_env import SumoEnv


# 注册环境
def create_env(env_config):
    return SumoEnv(config_path=env_config["config_path"])

register_env("sumo_env", create_env)


# 创建tensorboard记录文件
def custom_logger_creator(log_dir):
    def logger_creator(config):
 
        import time
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(log_dir, f"run_{timestr}")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(
            config=config,
            logdir=logdir,
          
        )
    return logger_creator


class PPOSimulation:
    def __init__(self, max_episode, config_file):
        self.max_episode = max_episode
        self.config_file = config_file
        # 配置 PPO 参数
        self.ppo_config = PPOConfig()
        self.ppo_config.environment(env="sumo_env", env_config={"config_path": config_file})
        self.ppo_config.exploration = {
            "type": "StochasticSampling",    # 使用 StochasticSampling 策略
            "stddev": 0.5,                   # 你可以设置探索时的标准差（可选）
        }
        self.ppo_config.observation_filter = "MeanStdFilter" 
        self.ppo_config.train_batch_size = 10000
        self.ppo_config.minibatch_size = 64
        self.ppo_config.gamma = 0.99
        self.ppo_config.lr = 1e-4
        self.ppo_config.num_workers = 16
        self.ppo_config.num_gpus = 1
        # 获取当前文件夹路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # 初始化日志系统
        self.log_file_path = os.path.join(self.current_dir, "logs", "ppo_simulation.log")
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            filename=self.log_file_path,
                            filemode="a",
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("PPO_Simulation")
       
        # 创建模型文件路径
        self.checkpoint_dir = os.path.join(self.current_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # 创建图片文件路径
        self.plot_dir = os.path.join(self.current_dir, "plot")
        os.makedirs(self.plot_dir, exist_ok=True)
        # 创建tensorboard文件路径
        self.board_dir = os.path.join(self.current_dir, "tensorboard_log")
        os.makedirs(self.board_dir, exist_ok=True)

        # 初始化 PPO 算法
        self.ppo_trainer = self.ppo_config.build(
            logger_creator=custom_logger_creator(self.board_dir)
        )

    def train(self):
        for episode in range(self.max_episode):
            result = self.ppo_trainer.train()
            
            # logging.info("Available metrics:", result.keys())
            # if "episode_reward_mean" in result:
            #     self.logger.info(f"Episode {episode} mean reward: {result['episode_reward_mean']}")
            #动态调整探索策略
            self.ppo_config.exploration["stddev"] = max(0.1, 0.5 * (1 - episode / self.max_episode))

            # 保存模型
            if episode % 10 == 0:
                checkpoint = self.ppo_trainer.save(self.checkpoint_dir)
                self.logger.info(f"Checkpoint saved at {checkpoint}")
            self.logger.info(f"回合========>>>>>>: {episode}")

    def test(self, checkpoint_path):
        # 恢复训练好的模型
        if hasattr(checkpoint_path, "local_path"):
            checkpoint_path = checkpoint_path["local_path"]
        self.ppo_trainer.restore(checkpoint_path)

        # 创建环境
        env = SumoEnv(config_path=self.config_file)
        state = env.reset()
        done = False
        total_reward = 0
        rewards = []

        while not done:
            action = self.ppo_trainer.compute_actions(state)
            print(f"action is: ====={action}")
            state, reward, done, _ = env.step(action)
            total_reward += reward
            rewards.append(total_reward)
        
        self.plot_rewards(rewards)
        return total_reward
    
    def plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward Curve")
        plt.savefig(os.path.join(self.plot_dir, "reward_curve.png"))


if __name__ == "__main__":
    # 使用绝对路径指定配置文件
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file = os.path.join(project_root, "config", "config.ini")

    # 每个task的训练回合数
    max_episode = 40
    # 启动 Ray
    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        #local_mode=True, 
        runtime_env={
            "working_dir": project_root,
            "py_modules": [
                os.path.join(project_root, "env")  # 包含自定义环境的路径
            ],
        },
    )

    ppo_sim = PPOSimulation(max_episode, config_file)
    ppo_sim.train()

    # 获取最佳检查点路径
    best_checkpoint = ppo_sim.ppo_trainer.save()
    logging.info(f"Best checkpoint saved at {best_checkpoint}")

    # 复现问题的代码，注释ppo_sim.train()，取消注释下面的代码即可
    # test_reward = ppo_sim.test(best_checkpoint)
    # logging.info(f"Test reward: {test_reward}")

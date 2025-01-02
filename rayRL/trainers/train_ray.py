import os
import sys
from pathlib import Path
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.sumo_env import SumoEnv
from config.config import SUMO_CONFIG  # 引用配置字典

# 注册环境
def create_env(env_config):
    return SumoEnv(config=env_config)

register_env("sumo_env", create_env)

# 创建tensorboard记录文件
def custom_logger_creator(log_dir):
    def logger_creator(config):
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(log_dir, f"run_{timestr}")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(
            config=config,
            logdir=logdir,
        )
    return logger_creator

class PPOSimulation:
    def __init__(self, max_episode, config):
        self.max_episode = max_episode
        self.config = config["sumo_env"]
        # 配置 PPO 参数
        self.ppo_config = PPOConfig()
        self.ppo_config.environment(env="sumo_env", env_config=self.config)
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
        """
        训练 PPO 模型
        """
        for episode in range(self.max_episode):
            result = self.ppo_trainer.train()
            
            # 动态调整探索策略
            self.ppo_config.exploration["stddev"] = max(0.1, 0.5 * (1 - episode / self.max_episode))

            # 保存模型
            if episode % 10 == 0:
                checkpoint = self.ppo_trainer.save(self.checkpoint_dir)
                self.logger.info(f"Checkpoint saved at {checkpoint}")
            self.logger.info(f"回合========>>>>>>: {episode}")
    
    def load_and_evaluate_model(self, checkpoint_path):
        """
        加载训练好的模型并进行评估
        """
        # 检查并提取路径字符串
        if isinstance(checkpoint_path, dict):
            checkpoint_path_str = checkpoint_path["local_path"]
        elif hasattr(checkpoint_path, "checkpoint") and hasattr(checkpoint_path.checkpoint, "path"):
            checkpoint_path_str = checkpoint_path.checkpoint.path
        else:
            checkpoint_path_str = checkpoint_path

        # 从检查点恢复训练器
        self.ppo_trainer.restore(checkpoint_path_str)

        module_path = Path(checkpoint_path_str) / "learner_group" / "learner" / "rl_module" / "default_policy"
        
        # 确保路径存在
        if not module_path.exists():
            raise ValueError(f"Module path does not exist: {module_path}")
            
        rl_module = RLModule.from_checkpoint(module_path)

        # 评估循环
        env = SumoEnv(config=self.config)
        obs, info = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        rewards = []
        
        while not (terminated or truncated):
            # 将observation转换为tensor并添加batch维度
            if isinstance(obs, dict):  # 对于多智能体环境
                actions = {}
                for agent_id, agent_obs in obs.items():
                    torch_obs = torch.from_numpy(np.array([agent_obs]))
                    # 获取动作分布输入
                    outputs = rl_module.forward_inference({"obs": torch_obs})
                    # 获取动作分布类
                    action_dist_class = rl_module.get_inference_action_dist_cls()
                    # 从logits创建动作分布
                    action_dist = action_dist_class.from_logits(outputs["action_dist_inputs"])
                    # 采样动作
                    action = action_dist.sample()[0].numpy()
                    actions[agent_id] = action
            else:  # 对于单智能体环境
                torch_obs = torch.from_numpy(np.array([obs]))
                outputs = rl_module.forward_inference({"obs": torch_obs})
                
                # 对于离散动作空间
                if hasattr(env.action_space, 'n'):  
                    action = torch.argmax(outputs["action_dist_inputs"][0]).numpy()
                # 对于连续动作空间
                else:  
                    action_dist_class = rl_module.get_inference_action_dist_cls()
                    action_dist = action_dist_class.from_logits(outputs["action_dist_inputs"])
                    action = action_dist.sample()[0].numpy()

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            rewards.append(episode_reward)
        env.close() # 关闭环境
        self.plot_rewards(rewards)
    
    def plot_rewards(self, rewards):
        '''
        绘制奖励曲线
        '''
        plt.plot(rewards)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward Curve")
        plt.savefig(os.path.join(self.plot_dir, "reward_curve.png"))

if __name__ == "__main__":
    # 使用配置字典中的配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = SUMO_CONFIG

    # 每个任务的训练回合数
    max_episode = config["sumo_env"]["max_episodes"]
    print(f"max_episode: {max_episode}")
    # 启动 Ray
    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": project_root,
            "py_modules": [
                os.path.join(project_root, "env")  # 包含自定义环境的路径
            ],
        },
    )

    ppo_sim = PPOSimulation(max_episode, config)
    ppo_sim.train()

    # 获取最佳检查点路径
    best_checkpoint = ppo_sim.ppo_trainer.save()
    logging.info(f"Best checkpoint saved at {best_checkpoint}")

    # 复现问题的代码，注释 ppo_sim.train()，取消注释下面的代码即可
    # ppo_sim.load_and_evaluate_model(best_checkpoint)
    # logging.info(f"Test reward: {test_reward}")
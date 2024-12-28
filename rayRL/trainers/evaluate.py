import torch
from ray.rllib.core.rl_module import RLModule
from pathlib import Path
import numpy as np
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.sumo_env import SumoEnv

 

def load_and_evaluate_model(checkpoint_path, env):
    """
    加载训练好的模型并进行评估
    
    Args:
        checkpoint_path: 模型检查点路径
        env: 评估环境实例
    """
    # 从检查点加载RLModule
    # 假设传入的是checkpoints目录的路径
    checkpoint_path = Path(checkpoint_path)
    module_path = checkpoint_path / "learner_group" / "learner" / "rl_module" / "default_policy"
    
    # 确保路径存在
    if not module_path.exists():
        raise ValueError(f"Module path does not exist: {module_path}")
        
    rl_module = RLModule.from_checkpoint(module_path)

    # 评估循环
    obs, info = env.reset()
    terminated = truncated = False
    episode_reward = 0.0
    
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

    return episode_reward

 
if __name__ == "__main__":
   
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config_file = os.path.join(project_root, "one_way_xml", "one_way.sumocfg")

    checkpoint_path = "E:/MyProjects/rayRL-----2024-11-23-/rayRL/trainers/checkpoints"
    
    env =  SumoEnv(
            render_mode="rgb_array",
            max_episodes=111,
            max_sim_time=8000,
            sumocfg=config_file,
        )
    
    # 运行评估
    total_reward = load_and_evaluate_model(checkpoint_path, env)
    print(f"Episode reward: {total_reward}")
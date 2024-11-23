import traci
import ray
from ray import tune
import argparse
from ray.rllib.algorithms.dqn import DQNConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pprint import pprint
from env.sumo_env import SumoEnv
from ray.tune.registry import register_env

def create_env(env_config):
    return SumoEnv(**env_config)

register_env("sumo_env",create_env)


# 使用无强化学习运行仿真
def run_no_rl(max_episode, config_file, task_id):
    print(f"Starting No-RL simulation for task {task_id}")
    env = SumoEnv(render_mode="rgb_array", max_episodes=max_episode, max_sim_time=5000, sumocfg=config_file)

    for episode in range(max_episode):
        state = env.reset()
        done = False
        while not done:
            traci.simulationStep()
            # action = env.action_space.sample()  # 随机动作
            # state, reward, done, _ = env.step(action)
            state = env._get_combined_state()
            done = env.check_terminated()
        print(f"Task {task_id}, Episode {episode} completed with random actions.")
    env.close()
    return f"Task {task_id}: No-RL simulation completed"

# 使用 DQN 算法运行仿真
def run_dqn(max_episode, config_file, task_id):
    print(f"Starting DQN simulation for task {task_id}")

    # 配置 DQN 参数
    dqn_config = DQNConfig()
    dqn_config.env = "sumo_env"
    dqn_config.env_config = {
        "render_mode": "rgb_array",
        "max_episodes": max_episode,
        "max_sim_time": 5000,
        "sumocfg": config_file,
    }
    dqn_config.gamma = 0.99
    dqn_config.lr = 1e-3
    dqn_config.train_batch_size = 64
    dqn_config.num_workers = 1  # 单线程训练
    dqn_config.num_gpus = 1  # 不使用 GPU，可根据需要调整

    # 初始化 DQN 算法
    dqn_trainer = dqn_config.build()

    # 开始训练
    for episode in range(max_episode):
        result = dqn_trainer.train()
        # print(f"Task {task_id}, Episode {episode}, Reward: {result['episode_reward_mean']}")
        # 保存模型
        if episode % 10 == 0:
            checkpoint = dqn_trainer.save()
            print(f"Task {task_id}, Checkpoint saved at {checkpoint}")
    return f"Task {task_id}: DQN simulation completed"

# Ray 远程任务：No-RL
@ray.remote(num_cpus=24)
def run_no_rl_remote(max_episode, config_file, task_id):
    return run_no_rl(max_episode, config_file, task_id)

# Ray 远程任务：DQN
@ray.remote(num_gpus=1)
def run_dqn_remote(max_episode, config_file, task_id):
    return run_dqn(max_episode, config_file, task_id)

if __name__ == "__main__":
    import os
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Select simulation mode")
    parser.add_argument("--mode", choices=["no-rl", "dqn"], default="dqn", help="Simulation mode: 'no-rl' or 'dqn'")
    parser.add_argument("--num_tasks", type=int, default=1, help="Number of tasks to run in parallel")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 使用绝对路径指定配置文件
    config_file = os.path.join(project_root, "sumo_xml", "three_points.sumocfg")
     
    max_episode = 200
    # 启动 Ray
    ray.shutdown()  #关闭之前的ray
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": project_root,  # 使用项目根目录作为工作目录
            "py_modules": [
                os.path.join(project_root, "env")  # 显式包含 env 模块
            ],
        },
    )
   
    
    #ray.init(ignore_reinit_error=True)
    # print("Ray initialized:", ray.is_initialized())
    # print("Current active workers:", ray.available_resources())

    # 根据选择的模式运行任务
    if args.mode == "no-rl":
        # 分配 No-RL 任务
        futures = [
            run_no_rl_remote.remote(max_episode, config_file, task_id=i)
            for i in range(args.num_tasks)
        ]
    elif args.mode == "dqn":
        # 分配 DQN 任务
        futures = [
            run_dqn_remote.remote(max_episode, config_file, task_id=i)
            for i in range(args.num_tasks)
        ]

    # 收集结果
    results = ray.get(futures)

    # 输出结果
    print("\nResults:")
    for result in results:
        pprint(result)
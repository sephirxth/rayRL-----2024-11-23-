import  gymnasium as gym
from gymnasium import spaces
import xml.etree.ElementTree as ET
import traci
import numpy as np
import os
from .traffic import TrafficSignalController #交通灯的类 


class SumoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        config
    ):
        """
        初始化 SUMO 环境。

        参数:
            render_mode (str): 渲染模式（'human(gui场景可见)' 或 'rgb_array(仅数据可见)'）。
            max_episodes (int): 最大回合数。
            max_sim_time (int): 每回合的最大仿真时间。
            sumocfg(str):动态仿真文件。
            sumocfg_out_flash(bool):是否输出文件。
        """
        super(SumoEnv, self).__init__()
        # 读取配置文件]
        self.render_mode = config["render_mode"]
        self.max_episodes = config["max_episodes"]
        self.max_sim_time = config["max_sim_time"]
        self.sumocfg = config["sumocfg"]
        self.sumocfg_out_flash = config["sumocfg_out_flash"]
        self.current_episode = 0 #运行回合计数
        self.simulation_running = False

        # 初始化交通信号灯控制器
        self.traffic_signals = []
        self.ts_controllers = {}

        # 动作和状态空间定义
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)  # 默认值，可以在后面根据实际情况调整
        self.action_space = spaces.Discrete(5)  # 默认值，也可以在后面调整
        


    def _get_current_traffic_signals(self):
        """
        获取当前所有交通灯的 ID。
        
        返回:
            list: 所有交通灯的 ID 列表。
        """
        ts_ids = traci.trafficlight.getIDList()
        return ts_ids

    def _initialize_traffic_signal_controllers(self):
        """
        初始化交通信号灯控制器。
        """
        self.traffic_signals = self._get_current_traffic_signals()
        # print("ts ids :",self.traffic_signals)
        for ts_id in self.traffic_signals:
            controller = TrafficSignalController(
                ts_id=ts_id,
                ts_lanes=['E0_0','E3_0'],
                num_phases=6,  # 交通灯绿灯相位为6
            )
            self.ts_controllers[ts_id] = controller

        # 定义整体的状态空间和动作空间
        state_dim = sum([ctrl.observation_space.shape[0] for ctrl in self.ts_controllers.values()])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32)
        action_space_size = sum([ctrl.action_space.n for ctrl in self.ts_controllers.values()])
        self.action_space = spaces.Discrete(action_space_size)


    def reset(self, seed=None, **kwargs):
        """
        重置仿真环境。

        返回:
            np.ndarray: 初始状态。
            dict: 额外信息。
        """
        if seed is not None:
            np.random.seed(seed)
        self.current_episode += 1
        
        if self.sumocfg_out_flash:
            self.modify_output_prefix(f'{self.current_episode}-')

        # 关闭之前的仿真
        if self.simulation_running:
            self.close()

        # 启动新仿真
        self._start_sumo()
        self.simulation_running = True
        self._initialize_traffic_signal_controllers()

        # 重置信号灯控制器状态
        for controller in self.ts_controllers.values():
            controller.currn = 0
            controller.time_since_last_phase_change = 0

        # 获取初始状态
        state = self._get_combined_state()
        info = {"episode": self.current_episode}  # 可以在这里添加更多信息
        return state, info
    
    def modify_output_prefix(self, new_prefix):
        """
        修改SUMO配置文件中的<output-prefix>标签的value属性。
        """
        cfg_file = self.sumocfg
        try:
            tree = ET.parse(cfg_file)
        except ET.ParseError as e:
            raise
        root = tree.getroot()
        output_prefix = root.find('.//output-prefix')
        if output_prefix is not None:
            output_prefix.set('value', new_prefix)
            tree.write(cfg_file)
        else:
            print("未找到<output-prefix>标签")


    def _start_sumo(self):
        """启动 SUMO 仿真。"""
        home = os.getenv("SUMO_HOME")
        if not home:
            raise EnvironmentError("SUMO_HOME 环境变量未设置。")
        sumo_binary = os.path.join(home, "bin/sumo-gui" if self.render_mode == "human" else "bin/sumo")
        sumo_cmd = [sumo_binary, "-c", self.sumocfg, "--start", "True", "--quit-on-end", "True","--no-warnings","True", "--no-step-log", "True", "--step-method.ballistic", "True" ]
        try:
            traci.start(sumo_cmd)
        except traci.exceptions.TraCIException as e:
            raise RuntimeError(f"启动SUMO失败,请检查sumo配置文件路径和SUMO安装: {e}")from e


    def _get_combined_state(self):
        """
        获取所有交通信号灯的联合状态。

        返回:
            np.ndarray: 状态向量。
        """
        states = []
        for controller in self.ts_controllers.values():
            states.append(controller.get_state())
        return np.concatenate(states)

    def step(self, action):
        """
        执行一个时间步。
        参数:
            action (int): 动作索引。
        返回:
            np.ndarray: 下一状态。
            float: 奖励。
            bool: 是否结束。
            dict: 额外信息。
        """
        # print("#################代码运行中#########################")
        # print(f"Action received: {action}")

        extra_reward = 0

        # 根据动作来设置不同的信号灯相位
        if self._can_switch_phase():
            if action == 1: 
                self.action(J1_phase=[0, 1], J2_phase=[0])  # J1相位0然后相位1，J2相位0
            elif action == 2: 
                self.action(J1_phase=[2], J2_phase=[1, 2])  # J1相位2，J2相位1然后相位2
            elif action == 3:  
                self.action(J1_phase=[3, 4], J2_phase=[3])  # J1相位3, 4, J2相位3
            elif action == 4:  
                self.action(J1_phase=[5], J2_phase=[4, 5])  # J1相位5, J2相位4, 5
            elif action == 0: #不切换相位
                extra_reward = 0
        else:
            extra_reward = 0 #无效情况

        # 进行仿真一步
        traci.simulationStep()

        # 检查是否结束
        terminated = (not traci.simulation.getMinExpectedNumber())
        truncated = (self.sim_step >= self.max_sim_time)

        # 计算奖励
        if action == 0:
            reward = extra_reward
        else:
            reward = self.reward()

        # 获取下一状态
        next_state = self._get_combined_state()
        info = {"step": self.sim_step, "reward": reward}
        return next_state, reward, terminated, truncated, info
        

    def action(self, J1_phase, J2_phase):
        """动作函数,设置每个节点应该对应的相位。"""

        # 延迟以确保车辆完全通过
        traci.simulationStep(5)
        for phase in J1_phase:
            traci.trafficlight.setPhase("J1", phase)

        for phase in J2_phase:
            traci.trafficlight.setPhase("J2", phase)

    def _can_switch_phase(self):
        """
        检查E1_0和E4_0车道是否没有车辆,确定是否可以安全切换相位。
        
        返回:
            bool: 如果E1_0或E4_0车道为空,True,则返回False。
        """
        # 检查E1_0和E4_0
        return traci.edge.getLastStepVehicleNumber("E0") == 0 and traci.edge.getLastStepVehicleNumber("E4") == 0

    def check_terminated(self):
        """仿真测试中断函数。"""
        done = self.sim_step >= self.max_sim_time or not traci.simulation.getMinExpectedNumber()
        # print(f"sim_step is {self.sim_step}")
        return done

    def reward(self):
        """
        计算所有交通信号灯控制道路上车辆平均速度延迟误差奖励。

        返回:
            float: 总奖励。
        """
        total_delay_error = 0.0
        total_vehicles = 0
        desired_speed = 14  # 期望速度（单位：m/s）
        max_delay_error = 14
        min_delay_error = 0

        # 获取所有车辆的ID列表
        vehicle_ids = traci.vehicle.getIDList()

        # 初始化一个列表来存储有效车辆的速度
        vehicle_speeds = []

        for vehicle_id in vehicle_ids:
            try:
                # 获取车辆当前速度
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                vehicle_speeds.append(current_speed)
            except traci.exceptions.TraCIException:
                # 如果车辆ID无效，跳过此ID
                continue

        # 计算延迟误差并累加
        delay_errors = np.abs(desired_speed - np.array(vehicle_speeds) )
        total_delay_error = np.sum(delay_errors)
        total_vehicles = len(vehicle_speeds)

        # 计算平均延迟误差
        avg_delay_error = total_delay_error / total_vehicles if total_vehicles > 0 else 0.0

        # 奖励计算，负的平均延迟误差
        reward = - avg_delay_error
        # normalized_reward =  2 * ((reward - min_delay_error) / (max_delay_error - min_delay_error)) - 1 # [-1,1]
        normalized_reward =  ((reward - min_delay_error) / (max_delay_error - min_delay_error)) # [0,1]

        return normalized_reward
        
    @property
    def sim_step(self):
        """
        当前仿真时间。

        返回:···
            float: 仿真时间。
        """
        return traci.simulation.getTime()

    def close(self):
        """
        关闭仿真。
        """
        if self.simulation_running:  # 仅在仿真仍在运行时关闭
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass  # 如果已经关闭，忽略异常
        self.simulation_running = False


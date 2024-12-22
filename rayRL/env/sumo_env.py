import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import os
from .traffic import TrafficSignalController  # 交通灯的类
from utils.state_builder import StateBuilder   
from utils.reward_calculator import RewardCalculator  
from utils.action_processor import ActionProcessor  

class SumoEnv(gym.Env):
    def __init__(self, config):
        """SUMO 环境 的 gym封装"""
        super().__init__()
        self.config = config
        self.current_episode = 0  # 运行回合计数
        self.simulation_running = False

        self.state_builder = StateBuilder()
        self.reward_calculator = RewardCalculator()
        self.action_processor = ActionProcessor()
      

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
                ts_lanes=["E0_0", "E3_0"],
                num_phases=6,  # 交通灯绿灯相位为6
            )
            self.ts_controllers[ts_id] = controller

        # 定义整体的状态空间和动作空间
        state_dim = sum(
            [ctrl.observation_space.shape[0] for ctrl in self.ts_controllers.values()]
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        action_space_size = sum(
            [ctrl.action_space.n for ctrl in self.ts_controllers.values()]
        )
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

        # 关闭之前的仿真
        if self.simulation_running:
            self.close()

        # 启动新仿真
        print(self.simulation_running)
        self._start_sumo()
        self.simulation_running = True
        self._initialize_traffic_signal_controllers()

        # 重置信号灯控制器状态
        for controller in self.ts_controllers.values():
            controller.current_phase = 0
            controller.time_since_last_phase_change = 0

        # 获取初始状态
        state = self.get_state()
        info = {"episode": self.current_episode}  # 可以在这里添加更多信息
        return state, info

    def _start_sumo(self):
        """启动 SUMO 仿真。"""
        home = os.getenv("SUMO_HOME")
        if not home:
            raise EnvironmentError("SUMO_HOME 环境变量未设置。")
        sumo_binary = os.path.join(
            home, "bin/sumo-gui" if self.render_mode == "human" else "bin/sumo"
        )
        sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumocfg,
            "--start",
            "True",
            "--quit-on-end",
            "True",
            "--no-warnings",
            "True",
            "--no-step-log",
            "True",
            "--step-method.ballistic",
            "True",
        ]
        traci.start(sumo_cmd)

    def get_state(self):
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
        else:
            print("Can not switch phase: E1 or E4 lane is not clear !!!")

        # 进行仿真一步
        traci.simulationStep()

        # 检查是否结束
        done1 = (
            self.sim_step >= self.max_sim_time
            or not traci.simulation.getMinExpectedNumber()
        )
        done2 = False

        # 计算奖励
        reward = self.reward()

        # 获取下一状态
        next_state = self.get_state()
        info = {"step": self.sim_step, "reward": reward}
        return next_state, reward, done1, done2, info

    def action(self, J1_phase, J2_phase):
        """动作函数。"""
        self._set_traffic_signals("J1", J1_phase)
        self._set_traffic_signals("J2", J2_phase)

    def _set_traffic_signals(self, ts_id, phases):
        """设置相位函数。"""
        for phase in phases:
            traci.trafficlight.setPhase(ts_id, phase)

    def _can_switch_phase(self):
        """
        检查E1_0和E4_0车道是否没有车辆,确定是否可以安全切换相位。

        返回:
            bool: 如果E1_0或E4_0车道为空,True,则返回False。
        """
        # 检查E1_0和E4_0车道上的车辆数量
        e1_0_vehicle_count = traci.edge.getLastStepVehicleNumber("E0")
        e4_0_vehicle_count = traci.edge.getLastStepVehicleNumber("E4")

        # 如果两个车道都没有车辆，则可以切换相位
        if e1_0_vehicle_count == 0 and e4_0_vehicle_count == 0:
            return True
        else:
            return False

    def check_terminated(self):
        """仿真测试中断函数。"""
        done = (
            self.sim_step >= self.max_sim_time
            or not traci.simulation.getMinExpectedNumber()
        )
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
        # 遍历每个交通信号灯
        for controller in self.ts_controllers.values():
            vehicle_ids = traci.trafficlight.getControlledLanes(controller.ts_id)

            for vehicle_id in vehicle_ids:
                if vehicle_id in traci.vehicle.getIDList():
                    # 获取车辆当前速度
                    current_speed = traci.vehicle.getSpeed(vehicle_id)

                    # 期望速度
                    desired_speed = 7

                    # 计算延迟误差并累加
                    delay_error = abs(current_speed - desired_speed)
                    total_delay_error += delay_error
                    total_vehicles += 1
                else:
                    continue

        # 计算平均延迟误差
        if total_vehicles > 0:
            avg_delay_error = total_delay_error / total_vehicles
        else:
            avg_delay_error = 0.0

        # 延迟误差越小奖励越高
        reward = -avg_delay_error
        return reward

    @property
    def sim_step(self):
        """
        当前仿真时间。

        返回:
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

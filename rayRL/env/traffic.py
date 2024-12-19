import numpy as np
import traci
from gymnasium import spaces

class TrafficSignalController:
    """控制单个交通信号灯的类，包含状态获取、动作应用和奖励计算的功能"""

    def __init__(self, ts_id, ts_lanes, num_phases, reward_fn=None):
        """
        初始化交通信号灯控制器

        参数:
            ts_id (str): 交通信号灯的ID
            ts_lanes(str): 交通灯实际控制的车道ID列表
            num_phases (int): 信号灯的相位数量
            reward_fn (callable, optional): 自定义奖励函数（如果没有提供，则使用默认函数）
        """
        self.ts_id = ts_id
        self.ts_lanes = ts_lanes
        self.num_phases = num_phases  

        # 初始化状态
        self.current_phase = 0
        self.time_since_last_phase_change = 0

        # 车道及其属性
        self.lanes = traci.trafficlight.getControlledLanes(self.ts_id)  # 获取该信号灯控制的车道
        self.filtered_lane = list(set(self.lanes) & set(self.ts_lanes))
        # print(f"ts_id is {self.ts_id},ts lane id {self.filtered_lane},current phase is {self.phase}")

        self.lane_lengths = {lane: traci.lane.getLength(lane) for lane in self.filtered_lane} # 获取车道长度信息

        # 定义动作空间和状态空间
        self.action_space = spaces.Discrete(num_phases)  # 动作对应绿灯相位的索引
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_phases + 2 * len(self.filtered_lane),),  # 只考虑锁定放行方向的车道
            dtype=np.float32,
        )

        # 奖励函数
        self.reward_fn = reward_fn or self.default_reward_fn  
        self.last_reward = 0

    def get_state(self):
        """
        获取当前交通信号灯的状态。

        返回:
            np.ndarray: 状态向量 [绿灯阶段one-hot编码, 是否超过最短绿灯时间, 车道密度, 车道排队比例]
        """
        # 当前绿灯阶段的 one-hot 编码
        self.current_phase = traci.trafficlight.getPhase(self.ts_id)
        # print(f"ts_id is {self.ts_id},current phase is {self.current_phase}")
        phase_one_hot = [1 if self.current_phase == i else 0 for i in range(self.num_phases)]
        
        # # 是否超过最短绿灯时间
        # min_green_passed = [1 if self.time_since_last_phase_change >= self.min_green else 0]

        # 获取车道密度和排队比例
        densities = [self._get_lane_density(lane) for lane in self.filtered_lane]
        queues = [self._get_lane_queue(lane) for lane in self.filtered_lane]

        # print(f"phase_one_hot is {phase_one_hot},densities is {densities},queues is {queues}")
        # 合并所有状态信息
        return np.array(phase_one_hot + densities + queues, dtype=np.float32)

    def _get_lane_density(self, lane):
        """计算车道的密度（车辆数 / 车道容量）"""
        return min(1.0, traci.lane.getLastStepVehicleNumber(lane) / self._get_lane_capacity(lane))

    def _get_lane_queue(self, lane):
        """计算车道的排队比例（排队车辆数 / 车道容量）"""
        return min(1.0, traci.lane.getLastStepHaltingNumber(lane) / self._get_lane_capacity(lane))

    def _get_lane_capacity(self, lane):
        """计算车道的容量（能够容纳的最大车辆数）"""
        return self.lane_lengths[lane] / (2.5 + traci.lane.getLastStepLength(lane))  # 假设车间距为2.5米


    def compute_reward(self):
        """计算交通信号灯的奖励"""
        self.last_reward = self.reward_fn()
        return self.last_reward

    def default_reward_fn(self):
        """默认奖励函数：负的所有车道的总等待时间"""
        waiting_times = [traci.lane.getWaitingTime(lane) for lane in self.filtered_lane]
        return -sum(waiting_times)
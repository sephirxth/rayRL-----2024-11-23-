SUMO_CONFIG = {
    "sumo_env": {
        "render_mode": "rgb_array",# "human",是否打开gui界面
        "max_episodes": 200,#最大训练回合数
        "max_sim_time": 18000,#最大仿真时间
        "sumocfg": "/home/jian/sumo/project/rayRL/one_way_xml/one_way.sumocfg",#sumo配置文件路径
        "sumocfg_out_flash": False  #是否输出sumocfg_output文件 T
        }
}
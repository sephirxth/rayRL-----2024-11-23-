#############文件描述#############
1.one_way.net.xml除去两边双向车道，中间单车道长度4400m，左侧道路长度150，右侧道路长度100m，仿真时间为6156s，定义基本红绿灯配时如下：
        <tlLogic id="J1" type="static" programID="0" offset="0">
            <phase duration="20" state="GG"/>
            <phase duration="365"  state="Gr"/> 

            <phase duration="385" state="Gr"/>

            <phase duration="20" state="GG"/>
            <phase duration="365"  state="Gr"/> 

            <phase duration="385"  state="Gr"/>
        </tlLogic>
        <tlLogic id="J2" type="static" programID="0" offset="0">
            <phase duration="385" state="rG"/> 

            <phase duration="20"  state="GG"/>
            <phase duration="365"  state="rG"/>

            <phase duration="385" state="rG"/>

            <phase duration="20"  state="GG"/>
            <phase duration="365"  state="rG"/>
        </tlLogic>
2.one_way.rou.xml定义基本车流量 50辆/时，定义车辆类型为卡车
3.one_way.sumocfg为仿真文件
4.data 中包含固定配时default 和 ppo策略的车辆信息输出文件，获取固定配时时运行test_local.py, ppo策略则运行train_ray.py
5.运行test_local.py获取车辆信息时，config文件中修改 "sumocfg_out_flash": True，则打开输出fcd文件配置，还需修改one_way.sumocfg中<tripinfo-output value="data/ppo/tripinfo/tripinfo-output.xml" />的目录为data/default/...，即可切换成功
6.切换到one_way_xml文件，其中data为输出的fcd文件，default中为固定配时的数据，运行xml2csv.py将其进行转换，运行plot_vehPerHour_LossTime.py则对数据进行处理，得到车流量-延误时间图。
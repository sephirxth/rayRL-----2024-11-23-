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
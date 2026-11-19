# Overleap：双自由度开源弹跳腿的官方实现

> 来源归档（repo / 官方一手源码）

- **中文标题：** Overleap：开源双自由度机械腿及弹跳控制
- **英文标题：** Overleap — Open-source 2-DOF Dynamic Robotic Leg
- **作者：** Aaron de los Santos（GitHub: aarondls）
- **官方仓库：** https://github.com/aarondls/overleap
- **代码：** https://github.com/aarondls/overleap/tree/main/Overleap
- **运行入口：** https://github.com/aarondls/overleap/blob/main/Overleap/examples/OverleapExample/OverleapExample.ino
- **源码主要文件：** [Overleap.cpp](https://github.com/aarondls/overleap/blob/main/Overleap/Overleap.cpp)、[Overleap.h](https://github.com/aarondls/overleap/blob/main/Overleap/Overleap.h)
- **许可：** MIT（仓库根目录与 Overleap 子目录均有 LICENSE）
- **开源状态：** 已开源（Arduino 控制库、例程、机械装配与零件资料公开）
- **入库日期：** 2026-11-19
- **核对范围：** 已阅读仓库 README 的 Overview、Design、Parts、Assembly、Programming、Single jump、Continuous hopping 等章节，Arduino 的 OverleapExample.ino 入口，以及 Overleap.cpp 与 Overleap.h 的相关实现（力曲线、关节位置 PD、触地判断、力矩映射、安全限幅、控制模式、标定与跳跃流程）。代码为静态核对，未实际编译烧录或实机复现。
- **通读：** 已完成
- **一句话说明：** Overleap 是 2-DOF 串联机械腿，通过 Arduino 经 UART 给 ODrive 发送电机侧力矩：空中使用关节 PD 保持姿态，蹬地时使用预设足端力曲线与雅可比转置生成力矩。

## 可核实实现

| 项目 | 源码位置与结论 |
|---|---|
| 几何 | 例程中构造 Overleap(0.19f, 0.22f, ...)；大腿 0.19 m、小腿 0.22 m |
| 传动 | Overleap.cpp 的 CalculateTorques 计算关节侧力矩后两轴均除以 9（理想 9:1） |
| 模式 | EnterTorqueControl 请求 CONTROL_MODE_TORQUE_CONTROL 与 CLOSED_LOOP_CONTROL |
| 竖直跳 | j：2500 N·ms；水平力上限 0；只跳一次 |
| 绕圈跳 | g：首次 2200 N·ms，之后 9 次 2600 N·ms；水平上限 10 N |
| 力曲线 | PrepareForceProfile 设 200 ms、b=0.005；UpdateForceProfile 设 a=冲量/100；ExecuteForceProfile 计算 a sin²(bπt) |
| PD 位置保持 | HoldPosition：目标角由上层传入，前 400 ms 对 Kp 线性爬坡，Kd 不爬坡 |
| 触地检测 | Touchdown 通过 analogRead(0) 读取 FSR 并判定触地，非连续足端测力 |
| 力矩传输 | ASCII 指令 c 0 / c 1 发送给 ODrive；轴 0 的雅可比映射力矩在执行阶段反号 |
| 安全 | SafeTorqueDirection 阻止向角度界外继续加速；出界时发送零力矩 |

## 精确读法和代码边界

- 正力的定义为足端 +X 向右、+Y 向下；角度输入三角函数前转换成弧度。
- 轨迹的“冲量”是期望力曲线的数学积分，**不代表测得地面反力冲量**。脚底 FSR 用于触地检测，不参与力的幅值闭环。
- 通过 9:1 减速比除法得到的是理想电机力矩，未在该段补偿传动损耗。
- 注释中的 1/250 与实际 200 ms、b=0.005 不一致，应以赋值和执行条件为准。
- ExecuteForceProfile 的离地提前退出判断设为 current_duration > 400，而 duration 固定为 200 ms；当前配置下不能触发该提前退出条件。仍可能因安全角方向检查提前中止。

## 对 wiki 的映射

- [Overleap 双自由度弹跳腿：分阶段 PD 与足端力映射](../../wiki/entities/overleap.md)
- [SLIP + VMC](../../wiki/methods/slip-vmc.md)（接触力映射的理论关联，非同一算法）
- [机器人雅可比矩阵](../../wiki/formalizations/robot-jacobian.md)

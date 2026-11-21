# DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation

> 来源归档（paper / arXiv:2510.11258v1）

- **URL：** <https://arxiv.org/abs/2510.11258>
- **PDF：** <https://arxiv.org/pdf/2510.11258.pdf>
- **项目页：** <https://beingbeyond.github.io/DemoHLM/>
- **代码仓库：** <https://github.com/BeingBeyond/DemoHLM>（`@cb913c1` 仅项目页与文档，无源码）
- **标题：** DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation
- **作者：** Yuhui Fu, Feiyang Xie, Chaoyi Xu, Jing Xiong, Haoqi Yuan, Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Peking University
- **年份：** 2025（arXiv 2025-10-13）
- **核对范围：** 已获取并通读 `arXiv:2510.11258v1` PDF 抽取全文（Sections 1–5 及 Appendix A–C，涵盖 IsaacGym Apple Vision Pro 单演示采集、Pink 多目标 IK、三阶段轨迹重放合成管线、AMO 50 Hz 全身底层 + 2-DoF 主动视觉比例跟踪、10 Hz 高层 BC 策略及 10 项仿真与 7 项 Unitree G1 真机测试）；未独立复现仿真与真机硬件部署。
- **通读：** 已完成

## 摘要与核心贡献

1. **单演示可泛化人形移动操作数据引擎**：针对真实全身遥操作难扩展的问题，仅通过 Apple Vision Pro 在 IsaacGym 中采集 **1 条人类参考演示**，按初次接触时刻 $t_c$ 切分为**接触前（物体相对坐标系 $T_{\text{obj}}^{\text{eef}}$）**与**接触后（本体相对坐标系 $T_{\text{pro}}^{\text{eef}}$）**两段；当机械人与物体位姿在预设区域 $R_1$–$R_3$ 随机化时，自动编排**Locomotion（PD 引导行走至近处）-> Pre-manipulation（插值衔接 + 物体相对轨迹）-> Manipulation（本体相对轨迹）**并经 Pink 多目标 IK 与底层控制器回放，自动合成百至五千条成功轨迹。
2. **分层控制与硬件改造**：
   - 底层通用全身控制器沿用 **AMO（50 Hz）**，接收低维高层指令 $(v_x, v_y, \omega, h, r, p, y, q_{\text{upper}})$ 并以 500 Hz Unitree SDK v2 PD 驱动 Unitree G1（29 DoF）；
   - 自制 **2-DoF 主动云台颈部**（Dynamixel XC330）+ **RealSense D435 RGB-D**（640×480 @ 60 Hz，按目标相机系位置 $(x_c, y_c)$ 做比例伺服跟踪）以及 3D 打印平行夹爪；
   - 高层策略对比 **MLP / ACT / Diffusion Policy（10 Hz）**，真机依靠 **FoundationPose / FoundationPose++（~25 Hz 异步跟踪）** 提供相机系物体 6D 位姿。
3. **仿真与真机表现**：10 项任务随合成数据从 100 增至 5,000 条呈单调提升（LiftBox 达 **98.83%**，PushCart **95.79%**，Handover **57.50%**，ACT 与 DP 显著优于 MLP）；7 项 Unitree G1 真机零样本迁移取得与仿真一致表现（LiftBox **5/5**、PressCube **5/5**、PushCube **4/5**、Handover **4/5**、GraspCube **3/5**、OpenCabinet **2/5**、EraseBoard **2/5**）。

## 与知识库关联

- 实体节点：[paper-loco-manip-161-136-demohlm.md](../../wiki/entities/paper-loco-manip-161-136-demohlm.md)
- 代码与项目页归档：[demohlm.md](../repos/demohlm.md)、[demohlm.md](../sites/demohlm.md)

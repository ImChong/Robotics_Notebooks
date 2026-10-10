# HuMBLE 来源归档

- **论文：** [HuMBLE: Human Motion-Driven Behavior Learning for Embodied Locomotion](https://arxiv.org/abs/2610.10489)
- **HTML：** https://arxiv.org/html/2610.10489v1
- **PDF：** https://arxiv.org/pdf/2610.10489
- **arXiv ID：** 2610.10489（cs.RO），v1 提交于 2026-10-07
- **作者与机构：** Mike Zhang 等；作者单位包括 RAI Institute、Boston Dynamics、Carnegie Mellon University
- **代码/项目页：** 论文未给出公开代码仓库或独立项目页；本次核查未找到作者关联的公开实现
- **数据：** 论文称约 3 小时、528,214 帧的人类步态数据。Boston Dynamics Atlas 数据因专有属性不能公开；G1 参考动作数据及其图表/表格底层数据“将发布”（论文原文为未来时，入库时不视为已发布）。

## 方法摘录

HuMBLE 试图同时保留人类步态风格和开放式速度指令跟踪能力。第一阶段以 retarget 后的人类全身参考轨迹训练 reference-conditioned teacher，再经 teacher–student / DAgger 式蒸馏得到只接收本体感知和 SE(2) 躯干速度指令的轻量策略。部署策略不需要运行时人体参考轨迹。

第二阶段从该先验继续做多任务 PPO 微调：短时域 reference-guided imitation 用于维持人类动作风格，长时域 goal-conditioned tracking 用于学习超出数据分布的指令覆盖与鲁棒性。goal-conditioned reference-state initialization（GC-RSI）按待执行速度指令，从相似的参考动作状态附近初始化 rollout；标称参考/目标环境分配为 50:50。

输入包括机载估计的线速度、IMU 角速度与重力投影、关节位置/速度、上一时刻动作，以及 `(v_{long}, v_{lat}, ω_{turn})) 指令。单一 MLP 以 50 Hz 输出关节目标，由关节级 PD 控制器跟踪。Unitree G1 上报告 Jetson AGX Orin CPU + ONNX Runtime 推理均值 1.1 ms；论文称 Boston Dynamics 平台推理延迟未披露（保密协议）。

## 数据与评测口径

- Vicon 动捕 120 Hz；动作经 BVH、两阶段全身 retarget、速度意图标注及左右镜像扩增。数据覆盖慢走、正常/快速行走、慢跑/冲刺、侧步、转向等平地步态。
- 在 Boston Dynamics Atlas R1、Atlas D1 与 Unitree G1 上验证，包含模拟与真机结果、用户摇杆控制、导航底层策略集成及箱体跑酷层级控制演示。
- Atlas R1 上，论文报告 HuMBLE 的大部分 SE(2) 命令区域跟踪误差与不使用人体数据的 Tabula Rasa RL 基线相近；多数 sim-to-real 测试的线速度与角速度误差差异分别在 0.05 m/s 与 0.10 rad/s 以内。Atlas R1 推扰实验在若干主方向上可承受最高 1400 N。
- 取舍：高速区间 HuMBLE 的响应更平滑、加速更渐进；快速斜向步行等数据稀疏区域的误差较大。纯 goal-conditioned 微调会损伤风格，纯 reference-guided 策略又会在 OOD 侧移等命令下失稳。

## 限制与开放状态

- 论文实验限于平地标准 locomotion；复杂地形需要额外地形观测与训练设计。
- 依赖 SE(2) 意图标注；更复杂的语义命令难以直接从轨迹计算。
- 论文没有给出 HuMBLE 训练代码或可下载策略权重；G1 数据为计划发布，Atlas 数据不可公开。复现与部署仍需自建仿真、机器人资产、动捕/retarget 数据和训练流水线。
- 作者尚未给出 MLP 容量扩展规律，或对 loco-manipulation、遥操作等任务的验证。

## 参考入口

- [arXiv 摘要与版本信息](https://arxiv.org/abs/2610.10489)
- [arXiv HTML 全文与补充材料](https://arxiv.org/html/2610.10489v1)
- [arXiv PDF](https://arxiv.org/pdf/2610.10489)

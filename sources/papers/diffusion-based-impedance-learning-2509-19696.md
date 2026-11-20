# Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks｜面向接触丰富操作任务的基于扩散模型的阻抗学习

> 论文原文归档（paper；同一论文仅创建一个 wiki 详情节点）

- **类型：** paper（IEEE Transactions on Robotics）
- **英文题名：** Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks
- **中文题名：** 面向接触丰富操作任务的基于扩散模型的阻抗学习
- **作者：** Noah Geiger, Tamim Asfour, Neville Hogan, Johannes Lachner
- **机构：** 卡尔斯鲁厄理工学院（KIT）Institute for Anthropomatics and Robotics / H²T；麻省理工学院（MIT）Department of Mechanical Engineering / Department of Brain and Cognitive Sciences
- **发表：** IEEE Transactions on Robotics, vol. 42, pp. 3160–3177, 2026；IEEE 日期 2026-07-27
- **DOI：** https://doi.org/10.1109/TRO.2026.3717255
- **arXiv：** 2509.19696（2025-09-24 v1；2026-08-11 v4）
- **原文：** https://arxiv.org/html/2509.19696v4
- **PDF：** https://arxiv.org/pdf/2509.19696
- **项目：** https://strokeairobotics.github.io/Diffusion-Based-Impedance-Learning_T-RO/
- **代码：** https://github.com/StrokeAIRobotics/Diffusion-Based-Impedance-Learning_T-RO
- **核对版本：** arXiv 2509.19696v4；2026-11-20 核对官方出版记录、项目主页及官方仓库
- **核对范围：** 已通读 v4 HTML 的摘要、I 引言、II 扩散与能量域阻抗控制基础、III sZFT 重建 / SLERP / 数据采集 / 模型选择 / 能量估计及算法 1、IV parkour/peg-in-hole 真机实验与消融、V 讨论、VI 结论、附录 A 的 peg-in-hole 图表与参考文献列表；并核查作者项目页和官方 README、Python 训练、部署 IPC、C++ 控制入口。未逐篇复现参考文献，未运行代码或真机实验。
- **通读：** 已完成
- **wiki：** [唯一论文详情](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md)

## 核心论文摘录（原文 → wiki）

### 1. 信息域与能量域的接口：预测 sZFT，不直接生成刚度

- 引言与 III-A 把名义零力轨迹（ZFT）定义为预设运动参考，把模拟零力轨迹（sZFT）定义为扩散网络依接触测量重建的平衡轨迹。
- Transformer 扩散模型利用外部六轴 wrench 的 cross-attention 与末端位姿，反推接触扰动下的平衡位姿；**名义 ZFT 在部署中仍保持为运动指令，sZFT 专供阻抗自适应**。
- 与仅输出动作的 Diffusion Policy 不同，网络不是直接回归关节力矩或刚度增益。
- **对 wiki 的映射：** [核心原理](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md#核心原理)

### 2. 几何一致的旋转扩散与分方向阻抗

- III-B2 采用 SLERP 在单位四元数空间注入/移除旋转扰动，避免对四元数逐元素加高斯噪声破坏单位范数。
- III-E 的能量估计通过接触 wrench 与重建位移建立弹性功关系；Algorithm 1 在 200 Hz 的 16 帧滑窗（80 ms）上按轴估计刚度削减项，使用 1 N / 1 Nm 阈值及数值正则。
- III-E4 根据 sZFT 表征的运动方向：任务相关轴维持高刚度，非任务方向更大幅降低刚度；阻尼与速度反馈一起进入估计/控制。
- **对 wiki 的映射：** [流程总览与模型](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md#流程总览)

### 3. 数据、精度与模型选型

- III-C 通过 Apple Vision Pro 遥操作采集 parkour 与机器人辅助上肢治疗动作，并同步记录 ATI Gamma 力/力矩。
- III-D Table II：选用隐藏维 512、4 个注意力头、6 层的 Transformer；parkour 的位置误差 0.994 mm、角度幅值误差 0.249°、旋转轴误差 0.003°。
- 两类数据混合共 72,000 样本，文章称相当于 1.6 h 的 200 Hz 示教；混合训练后位置误差 0.883 mm、角度幅值误差 0.233°，旋转轴误差 0.002°。
- **对 wiki 的映射：** [实验与评测](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md#实验与评测)

### 4. 真机实验与关键消融

- IV 采用 7 DoF KUKA LBR iiwa，FRI 5 ms 力矩控制，初始最大刚度：平移 800 N/m、转动 150 Nm/rad；内置重力/科氏等补偿。
- Parkour：固定高刚度在第一障碍物卡住并触发速度保护；自适应模型完成全部障碍（限速 0.24 m/s、接触力保护阈值 20 N）。
- IV-C Table III 每类 peg 30 次；完整方法圆柱/方形/星形均 30/30。固定高刚度分别 30/30、4/30、0/30；手动最佳非均匀参数为 30/30、26/30、21/30；“仅 sZFT 不做方向性”三类皆 0/30，“仅从名义 ZFT 做方向调节”仅圆柱 30/30。
- **对 wiki 的映射：** [评测与方法对比](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md#与其他工作对比)

## 原文局限与工程边界

- V–VI 说明单臂、单平台、固定名义 ZFT；未验证双臂协作、高动态任务或大规模跨平台迁移。
- 测试集不包括 peg 专用演示，证明本实验设置下的跨任务表现；不等于所有几何装配零样本成功。
- 接触安全依赖 F/T 标定、扭矩控制、刚度/阻尼限幅及外部硬件保护，论文结果不能替代工业级安全验证。
- 官方 README 提供训练脚本及 C++ 控制代码，但部署需要配置 KUKA FRI、传感器、模型路径与共享内存；未有本仓硬件复现结果。

## 归档关联

- [项目页入口与视频](../sites/diffusion-based-impedance-learning.md)
- [源码、依赖及运行入口](../repos/diffusion-based-impedance-learning-tro.md)
- [论文唯一节点](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md)

## 引用

Geiger, N., Asfour, T., Hogan, N., & Lachner, J. (2026). Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks. IEEE Transactions on Robotics, 42, 3160–3177. https://doi.org/10.1109/TRO.2026.3717255

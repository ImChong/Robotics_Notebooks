# ALOHA / ACT 论文通读（遥操作硬件 + 示范学习视角）

> 来源归档（原文通读 · 支撑 `wiki/tasks/teleoperation.md`）

- **标题：** Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- **作者：** Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn（Stanford / UC Berkeley / Meta）
- **类型：** paper / teleoperation / imitation-learning
- **原文：** <https://arxiv.org/abs/2304.13705>（2026-11 通过 arXiv PDF v1 获取全文，RSS 2023）
- **一句话说明：** 提出不到 2 万美元的双臂主从遥操作系统 ALOHA，以及用动作分块（action chunking）+ CVAE 的 Transformer 策略 ACT，用约 10 分钟示范学会精细双臂操作。
- **核对范围：** 通读正文 §I–§VII（系统设计、ACT 方法、8 个任务与基线对比、三组消融与 5Hz/50Hz 用户研究、局限）及附录 A–F（成本对比、观测图像、网络结构、超参表、用户研究、局限）；参考文献列表仅浏览。未阅读 ALOHA 2、Mobile ALOHA 等后续工作，也未阅读本页其余来源（XR 全身、UMI 系等）的原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **系统设计原则（§III）：** 低成本、通用、易用、可维修、易搭建；两台 ViperX 6-DoF 从臂（约 5600 美元/台，负载 750 g，精度 5–8 mm）+ 小一号 WidowX 作主臂，平行夹爪 + 3D 打印"透明"指尖，整套 <20k 美元；不用灵巧手（成本与维护）。
- **关节空间映射（§III）：** 操作员反向驱动主臂，从臂关节同步，而非 VR 手柄的末端位姿映射。理由：精细操作常在奇异点附近而 6-DoF 无冗余，现成 IK 常失败；关节映射带宽高、算量小、延迟低；主臂自重抑制快速动作和抖动。另有"手柄—剪刀"机构（连续控制夹爪、降低反驱力）与橡皮筋重力平衡（可持续 >30 分钟遥操作）。
- **采集与观测：** 4 个 Logitech C922x 相机（2 腕部 + 前 + 顶，480×640），遥操作与记录均 50 Hz；动作取主臂关节位置（而非从臂），因为力大小由两者差经底层 PID 隐式决定；观测为从臂关节 + 4 路图像；动作空间 14 维绝对关节位置。
- **动作分块（§IV-A）：** 每 k 步观测一次并输出 k 步动作，有效时域缩小 k 倍，缓解模仿学习的复合误差，并能建模示范中的停顿等非马尔可夫行为；时间集成（temporal ensemble）每步都查询策略，对同一时刻的多个预测按 w_i=exp(−m·i) 加权平均，使动作平滑，无额外训练代价。
- **CVAE（§IV-B）：** 人类示范有随机性，策略训练为条件 VAE（编码器 BERT 式，仅训练时用，输入关节 + 动作序列，不含图像；测试时 z 取先验均值 0）；重构用 L1，正则项权重 β；L1 优于 L2，目标关节位置优于差分关节位置。
- **实现（§IV-C，附录 C/D）：** ResNet18 图像特征 + 2D 正弦位置编码，Transformer 编码器（4 层）/ 解码器（7 层，cross-attention），k=100，β=10，lr 1e-5，batch 8，约 80M 参数，每任务从头训练，单张 2080 Ti 约 5 小时，推理约 0.01 s。
- **任务与数据（§V）：** 6 个真机任务（Slide Ziploc、Slot Battery、Open Cup、Thread Velcro、Prep Tape、Put On Shoe）+ 2 个 MuJoCo 仿真任务；每任务 50 条示范（Thread Velcro 100 条），单条 8–14 s（400–700 步），共约 10–20 分钟数据。
- **结果（§V-C，表 I/II）：** ACT 在真机 Slide Ziploc 88%、Slot Battery 96%、Open Cup 84%、Prep Tape 64%、Put On Shoe 92%、Thread Velcro 20%；对比 BC-ConvMLP、BeT、RT-1、VINN，基线在真机上多停在第一阶段（BeT 在后三个任务最终成功率为 0）。
- **消融（§VI）：** 分块大小 k 从 1 到 100，4 种设置平均成功率 1%→44%，k 更大略降；BC-ConvMLP、VINN 加分块同样受益；时间集成对 ACT +3.3%、BC-ConvMLP +4%，对 VINN 反而下降；人类示范下去掉 CVAE 成功率 35.3%→2%，脚本示范则几乎无差别。
- **用户研究（§VI-C，附录 E）：** 6 名无 ALOHA 经验的研究生，穿扎带 / 拆杯子两任务，5 Hz 对 50 Hz 遥操作耗时增加 62%（33 s vs 20 s；16 s vs 10 s），p<0.001。
- **成本对比（附录 A）：** DexPilot 类约 10 万美元，单目相机手部重定向类约 1.8 万美元（单臂+手），Shadow 双手遥操作系统 ≥40 万美元；ALOHA 18k–20k 美元，并复现了 Shadow 演示视频的 15 个任务中的 14 个（缺手内旋转 Baoding 球，因无灵巧手）。
- **局限（§VII，附录 F）：** 硬件上做不了需多指协同（儿童锁药瓶）、大力矩（拧开密封水瓶、提重物）、需指甲（撕胶带边、开易拉罐）的任务；学习上拆糖纸（10 次 0 次成功，感知与数据不足）和桌面平放小号密封袋（中空步骤失败）未学会；缺扣衬衫扣子同样超出能力。

## 对 wiki 的映射

- [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2304.13705>

# NEXUS 论文来源归档

## 元数据

- **题目：** NEXUS: Perceptive Whole-Body Control for Terrain-Adaptive Teleoperation
- **作者：** Xiangyu Miao, Junsong Wu, Jiyuan Shi, Weiji Xie, Jinrui Han, Xingyi Wang, Weinan Zhang†, Chenjia Bai†, Xuelong Li（† 通讯作者）
- **机构：** 中国电信人工智能研究院（TeleAI）、上海交通大学、香港大学、浙江大学、γ-Robotics（伽马智能）
- **预印本：** https://arxiv.org/abs/2609.39000
- **项目页：** https://nexus-humanoid.github.io/（页内 Paper / arXiv / Code 按钮由 `static/js/site-config.js` 填充，Code 指向官方仓库）
- **代码：** https://github.com/Gamma-Robot/NEXUS（文档 https://gamma-robot.github.io/NEXUS/）
- **版本：** v1，2026-09-30，cs.RO；截至 2026-11-06 arXiv 仅此一版
- **核对范围：** 2026-11-06 通读 arXiv:2609.39000v1 PDF 全文 12 页：正文 §I–V（含方法 §III-A–C、实验 §IV-A–D）、参考文献、附录 A–C（运动适配参数表 III、奖励表 IV、域随机化表 V、超参与网络表 VI、评测协议）；图 4 数值按 300 dpi 渲染后读柱标。另核对项目页资源配置，以及官方仓库 commit `f390cb7`（2026-10-29）的 README、`docs/guides/`、`resources/g1/checkpoints/README.md` 与 `train.py` / `play.py` / runner / 任务注册入口。
- **通读：** 已完成
- **入库日期：** 2026-11-06
- **一句话说明：** 用接触引导的离线运动学适配为平地动作批量生成逐帧对齐的地形版本，再以特权教师 PPO + DAgger 学生蒸馏，得到只需平地命令、本体历史和单帧深度的 G1 地形适配全身遥操作控制器。

## 论文要解决的问题

全身遥操作中操作者在平地上运动，机器人却可能位于楼梯、斜坡或平台边缘；逐帧复制会产生错误接触。学习地形适配需要「同一行为在平地与地形上」的配对监督，而现有平地动作库没有地形版本，直接采集需额外动捕和场景重建，难以按行为 × 地形扩展。已有感知运动跟踪 PMT 的参考合成（TCRS）计算昂贵、质量不稳定，且 PMT 在其语料之外的行为上覆盖不足（§I–II）。

## 方法摘录

1. **地形条件运动适配（§III-B，附录 A）**：对脚踝、手腕连杆按高度、水平速度、竖直速度做带滞回的接触检测（进入 / 退出阈值 0.18 / 0.25 m 等，表 III）；向下射线查询地形高度与法向，骨盆按活动接触中绝对值最大的高度偏移整体平移；在接触关键帧上对每条活动肢体做解析两连杆 IK + 有界最小二乘（姿态残差权重 λ=0.08），脚底另按地形法向对齐（上限 40°）；随后做手 / 脚及膝 / 肘的穿透修正（弯曲平面旋转最多 45°、每方向 12 档、最多 3 个候选）；最后把关键帧上的「相对源动作修正量」用 PCHIP 插值到整段，非接触肢体基本保持原姿态。所有动作与地形共用一套参数。
2. **特权教师（§III-C）**：PPO 跟踪适配后参考 `M_adapt`，观测含参考窗口、特权本体与接触信息、11×17 根坐标高度图；奖励 = 跟踪 + 接触（时序、离地间隙、脚底支撑）+ 正则 + 存活（表 IV）；终止条件为根离地高度或末端相对高度偏差 > 0.25 m、重力投影 z 分量偏差 > 0.8；带左右对称损失与按失败率的自适应采样（50 帧分箱，混入 0.1 均匀分布）。
3. **感知学生**：DAgger 式行为克隆，输入为配对的**平地**命令 `m_flat`、10 帧本体历史和单帧深度图；三层 CNN（32/32/64 通道，全局最大池化）+ MLP [1024, 1024, 512, 512, 256]。部署时不需要适配后参考。
4. **动作接口**：动作为相对标称关节角的缩放偏移，PD 跟踪；PD 增益、电枢、力矩上限与动作尺度沿用 BeyondMimic（附录 B）。

## 论文报告的评测

- **训练设置**：mjlab，4 张 H100，每卡 16,384 并行环境；训练地形为平地、坡度 0.30（rise/run，约 16.7°）的上下坡、0.10 m 级高 / 0.30 m 踏面的上下楼梯；平地动作取自 SEED、LaFAN1、AMASS、100STYLE 子集，含地形版本合计近 1,000 小时。
- **运动适配（表 I；55 条 G1 动作 × 16 种地形，共 880 对，生成 39.87 h）**：无手接触组 NEXUS VTR 92.08%、CP 79.12%，TCRS 为 89.69% / 29.63%；有手接触组 NEXUS 80.53% / 71.21%，TCRS 67.95% / 20.61%。生成 RTF 比 TCRS 低 4.88 倍和 4.23 倍；Root-only 更快，但 VTR 仅 63.37% / 55.69%。无手接触组 TCRS 穿透更低（0.606 vs 0.814 cm），但悬空更大（2.965 vs 0.880 cm）。
- **控制器对比（图 4；160 条公开评测动作，2.61 h；对比 SONIC v1.1、HoloMotion、HumanoidGPT、PMT 公开 checkpoint）**：平地平均 SR NEXUS 90%，SONIC 88.33%，PMT 23.33%；易地形（5 cm 台阶 / 8.5° 坡）77.36% vs 次优 64%；难地形（15 cm / 24.2°）51.9% vs 次优 31.76%。PMT 在 100STYLE、LaFAN1 子集平地 SR 为 0。NEXUS 的 MPKRE 与足接触 F1 在平地和地形上均最好；地形 MPKPE 为 10.15 cm，高于 SONIC / HoloMotion / HumanoidGPT（9.79 / 8.83 / 9.29 cm），论文提醒误差只在各自失败前窗口内统计，须与 SR 一起看。
- **消融（表 II；415 条留出行走片段 × 4 种地形 × 5 级，8,300 次试验，20.01 h）**：SR 91.072 ± 14.959%，去掉视觉 87.771 ± 19.362%，去掉教师—学生（单阶段 PPO）90.964 ± 15.938%；后者 Root Pos 31.951 cm、MPKPE 9.482 cm，明显差于 NEXUS 的 19.596 / 7.066 cm。较难两级上视觉带来的 SR 差距为 78.10% vs 70.51%。± 为五个地形等级均值间的标准差，不是多种子方差。
- **真机（§IV-D）**：Unitree G1 零样本部署；机载 RealSense D435i 深度 60 Hz、480×270，裁边 2 px 后缩到 32×18，截断 0–2 m 并归一化；Jetson Orin NX 上 ONNX Runtime 推理，控制 50 Hz；人体姿态在另一台 PC 用 GMR 重定向，经 ZeroMQ 异步发送，命令窗口为 1 当前 + 6 历史 + 6 前瞻帧（前瞻 120 ms），丢帧时重复最新姿态。展示 15 cm 台阶连续上楼、从 60 cm 平台下台阶 / 跳下、平台边缘与草坡侧手翻、约 30° 草坡、单 / 双膝跪、单腿平衡、踢腿等；常规跟踪未见失败，60 cm 下落与侧手翻有摔倒。未给出真机成功率统计。

## 局限（论文与核对所见）

- 论文自述：适配算法不支持悬垂（overhanging）地形；穿透修正是局部搜索，不保证无碰撞（附录 A）；手部按胶囊体建模，不做法向对齐。
- 训练地形只有单一坡度和单一台阶高度；真机只有定性展示。
- 消融只在行走片段上做；单阶段 PPO 与蒸馏的预算不同（GPU 显存限制）。

## 开源状态与边界

截至 2026-11-06：项目页 Code 按钮指向 [Gamma-Robot/NEXUS](https://github.com/Gamma-Robot/NEXUS)。仓库（Python 3.13，依赖 mjlab 1.5.2、rsl-rl-lib 5.4.0）README 的 Release Status 勾选了独立地形运动适配包、策略训练管线、G1 预训练 ONNX 模型（`resources/g1/checkpoints/policy.onnx`，输入 `obs (1,1473)` 与 `camera (1,1,18,32)`，输出 29 维动作，观测构造与深度预处理在 ONNX 图外完成）和文档；**部署（sim-to-sim 与 sim-to-real）未勾选**。训练示例为 4 卡 × 16,384 环境。仓库只带 3 段示例动作（walk、crawl_and_back_handspring、backward_roll）× 平地与四种金字塔地形，未发布论文所用近 1,000 小时配对语料。`play.py` 回放 `.pt` checkpoint，可附带导出 ONNX，仓库内没有直接运行 `policy.onnx` 的脚本。许可：原创部分 CC BY-NC 4.0（非商业），RSL-RL 派生的 PPO 与蒸馏模块保留 BSD-3-Clause。

## 对 wiki 的映射

- [NEXUS 论文实体页](../../wiki/entities/paper-nexus-terrain-adaptive-teleoperation.md)
- [较早 NEXUS 预告页](../../wiki/entities/nexus-humanoid.md)（同一项目页 URL 的 2026-09-06 状态）

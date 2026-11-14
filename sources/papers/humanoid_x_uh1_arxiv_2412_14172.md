# Learning from Massive Human Videos for Universal Humanoid Pose Control（Humanoid-X / UH-1 原文通读归档）

- **标题：** Learning from Massive Human Videos for Universal Humanoid Pose Control（arXiv:2412.14172v1；IEEE-RAS Humanoids 2025 录用版本题为 *Universal Humanoid Robot Pose Learning from Internet Human Videos*，DOI: `10.1109/Humanoids65713.2025.11203143`）
- **作者与机构：** Jiageng Mao, Siheng Zhao, Siqi Song, Tianheng Shi, Junjie Ye, Mingtong Zhang, Yue Wang（University of Southern California）；Haoran Geng, Jitendra Malik（UC Berkeley）；Vitor Guizilini（Toyota Research Institute）
- **论文：** <https://arxiv.org/abs/2412.14172>（纠正：旧索引条目曾误注为 `arXiv:2501.05098`，实为 `Motion-X++` 的编号）
- **项目主页：** <https://usc-gvl.github.io/UH-1/>
- **数据集与模型入口：** <https://huggingface.co/datasets/USC-GVL/Humanoid-X> · <https://huggingface.co/USC-GVL/UH-1>
- **官方仓库：** <https://github.com/sihengz02/UH-1>
- **核对范围：** 通读 `humanoid_x_uh1_2412_14172.txt`（正文 §1–§6 及附录 A–E 全篇，含 Table 1–4 与 Appendix Table 1–5）；同步对照 `sihengz02/UH-1` 仓库。
- **通读：** 已完成

## 一句话摘要

提出从大规模互联网人类视频自动挖掘、标注、估计 3D 姿态、重定向到人形机器人关键点并用目标条件 RL 策略产出物理可执行目标关节角的 **Humanoid-X** 数据集（163.8K 片段、20.7M 帧、240.3 小时、5 种模态），并基于该数据集训练自回归 Transformer 模型 **UH-1** 实现文本到人形姿态与动作的通用控制（在 Unitree H1-2 上完成真机部署）。

## 原文核心事实核对

### 1. 数据集规模与来源分布（§3.1, Appendix B.1, Appendix Table 1）

- **总规模：** **163,800** 条（163.8K）单人动作样本、**20,700,000** 帧（20.7M，摘要写作 over 20 million）、总时长 **240.3 小时**；全部文本描述去重词表共 **11,897** 词（动词 3,206、名词 6,048、形容词 1,526、副词 590、其他 527；Appendix B.1 文本所提 `3206 words` 与 Table 5 动词词表大小一致）。
- **数据来源构成（Appendix Table 1）：**
  - 视频理解/在线视频：`Kinetics700`（68.6K clips / 5.2M frames / 72.4h）、`MotionX Video`（40.6K / 7.9M / 72.9h）、`Online Video`（YouTube Data API v3 按 400+ 搜索词检索：17.8K / 2.3M / 32.6h）、`Charades`（9.3K / 1.0M / 1.0h）、`HAA500`（5.2K / 0.3M / 2.9h）；
  - 动捕与数字人来源：`AMASS`（13.4K / 2.0M / 27.4h）、`IDEA400`（12.5K / 2.6M / 24.0h）、`AIST`（1.5K / 0.3M / 3.2h）、`GRAB`（1.3K / 0.4M / 3.8h）、`EgoBody`（1.0K / 0.4M / 4.0h）、`HuMMan`（0.7K / 0.1M / 1.0h）。

### 2. 五模态数据管线 `<V, T, P_human, P_robot, A_robot>`（§3.2, Appendix B–C）

1. **视频挖掘与单人动作筛选（`V`）：** 标准化下采样至 **20 fps**；使用 YOLOv8 检测仅含**恰好 1 名可见人**的帧，提取裁剪灰度 ROI 帧间差分与批次运动滤波，要求至少连续 **64 帧**满足显著运动条件。绝大多数片段短于 10 秒。*（伦理声明 Appendix A：为保护隐私与版权，对人物面部匿名化且不公开发布原始互联网视频 `V`。）*
2. **动作导向视频标注（`T`）：** 使用 **Video-LLaMA / VideoLLaMA 2** 对每段视频均匀采样 8 帧生成单句动作描述（`.txt`），Prompt 显式禁止描述人物外貌与服装，多数句子少于 20 词。
3. **3D 人体姿态估计（`P_human`）：** 用 **VIBE**（仓库文档亦提及 4DHumans）估计每帧 **SMPL** 参数 $(\beta, \theta, t_{\text{root}})$ 并结合弱透视相机缩放 $s$ 与预设焦距 $f$ 恢复 3D 根平移，存储为 **20 fps** 的 `.npy`。
4. **人到人形重定向（`P_robot`）：** 选取人类与人形机器人共有的 **12 个关键关节**（左右髋、膝、踝、肩、肘、腕）：先在标准 T 姿下以 Adam 优化 SMPL 体型参数 $\beta$（约束 $|\beta_i| < 5$）对齐机器人连杆尺度，再用 Adam 求解目标机器人 DoF 位置 $q_{\text{robot}}$（L1 关键点位置损失 + 权重 $\lambda=0.05$ 的二阶差分时间平滑项）。每帧存储为 **34 维向量**（27 维 H1-2 无手 DoF 关节角 + 3 维根位置 + 4 维根四元数），**20 fps** `.npy`。
5. **目标条件 RL 控制策略动作（`A_robot`）：** 在 **Isaac Gym**（6,192 并行环境，PPO 训练，仅用 **AMASS 的 CMU MoCap 子集**剔除人-物/复杂地形交互片段）训练目标条件跟踪策略 $\pi: \mathcal{G} \times \mathcal{O} \to \mathcal{A}_{\text{robot}}$；动作 $\mathcal{A}_{\text{robot}}$ 为 **27 维目标 PD 关节位置**，以 **50 fps** `.npy` 存储。为保护硬件脆弱踝滚转关节，两个 ankle roll 动作每步置零。

### 3. UH-1 模型、实验与局限（§4–§6, Table 1–4, Appendix D–E）

- **UH-1 架构：** VQ-VAE **Action Tokenizer**（下采样率 $k=4$，词表 $2048 \times 512$，一阶差分损失 + 根位置正则）+ 18 层 **UH-1 Transformer**（16 头，隐藏维 1,024，CLIP 文本特征条件自回归生成离散动作 token 并经 `[End]` 终止）。支持两种部署模式：**Text-to-Keypoint**（生成 `P_robot` 驱动闭环 RL 策略）与 **Text-to-Action**（直接输出开环上肢 `A_robot` 配合独立下肢行走策略）。
- **核心量化结果：**
  - **HumanoidML3D 基准（Table 1）：** UH-1（FID `0.445 ± 0.078`，MM Dist `3.249 ± 0.016`，R-Precision Top-3 `0.761 ± 0.003`）优于先生成人运动再重定向的两阶段基线 `MDM`（FID `0.582`）与 `T2M-GPT`（FID `0.667`）。
  - **Scaling 实验（Table 2 & Fig. 7）：** 在 Humanoid-X 预训练再微调将 HumanoidML3D 上 FID 从 `0.445` 降至 `0.379`；训练集比例从 1% 扩至 100% 使 FID 从 `0.689` 降至 `0.463`、Diversity 从 `5.900` 升至 `6.149`。
  - **真机评测（Unitree H1-2，Table 3）：** 12 项语言指令各测 10 次，**Text-to-Keypoint 平均成功率 94.33%**（含 9 项 100%，Cross Arms 80%，Boxing / Golf Putt 90%），**Text-to-Action 平均成功率 89.83%**。
- **作者自述局限（§6 & Appendix E）：** 论文仅研究姿态控制（pose control），明确指出**不包含物体交互与人形操作（humanoid manipulation / loco-manipulation）**；真机上 Text-to-Action 开环模式仅适用于上肢动作且需依赖独立下肢行走策略维持平衡。

## 知识库映射

- [Humanoid-X（BFM 行为数据）](../../wiki/entities/dataset-bfm-humanoid-x.md)
- [UH-1 官方代码库核对归档](../repos/uh1_humanoid_x.md)

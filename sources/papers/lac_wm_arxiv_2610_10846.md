# LAC-WM 论文来源归档

## 元数据

- **题目：** Cross-Embodiment Robot Foundation World Models with Latent Actions
- **方法：** LAC-WM（Latent Action-Conditioned World Model）；对照 EAC-WM（Explicit Action-Conditioned World Model）
- **作者：** Huang Huang、Sriram Yenamandra、Arjun Majumdar、Elie Aljalbout、Tushar Nagarajan、Tsung-Yen Yang、Akshara Rai、Michael Rabbat、Li Fei-Fei、Jiajun Wu、Tingfan Wu、Franziska Meier
- **单位：** Stanford University；部分工作在 Meta FAIR Robotics 完成（项目页）；项目页标注 ICML 2026
- **预印本：** https://arxiv.org/abs/2610.10846
- **HTML：** https://arxiv.org/html/2610.10846v1
- **版本：** v1，2026-10-07，cs.RO
- **项目页：** https://lacwm.github.io/ （归档见 [sources/sites/lacwm-github-io.md](../sites/lacwm-github-io.md)）
- **代码：** 论文与项目页均未列出代码仓库（截至 2026-10-12）

## 论文要解决的问题

不同机器人的关节结构与动作空间不同。世界模型若直接以各机器人的显式动作为条件，每个本体会学到彼此分离的动作表示，难以把一种机器人学到的动力学迁移到另一种。论文用学习得到的共享潜在动作空间统一不同本体。

## 方法与训练信息

1. **视觉编码：** 冻结 V-JEPA 2（1B）RGB 分词器，256×256 图像得到 16×16 个 1408 维 token。
2. **逆动力学 IDM：** 4 个因果注意力块、16 头、隐维 512（约 4694 万参数）；一个可学习 query token 对帧嵌入做 cross-attention，输出连续潜在动作 z_t（动作嵌入维 64）。
3. **前向动力学 FDM：** 8 块、16 头、隐维 512（约 9384 万参数）；每块含帧 token 自注意力、动作自注意力、帧对动作的 cross-attention；预测下一帧嵌入。
4. **运动解码器 MD：** z_t 通过 cross-attention 查询当前帧视觉 token，再接 MLP 输出运动标签；潜变量一半解码末端位姿，一半解码相机位姿。作为辅助监督，防止捷径解。
5. **共享潜空间：** 人与机器人数据端到端联合训练；cross-augmentation 让 IDM 看一种增广视图、FDM 从另一种视图预测。损失 L = λ_recon·L_recon（预测下一嵌入的平方 L2）+ λ_motion·L_motion（运动标签 MSE），λ 值原文未给出。
6. **动作投影器（微调时）：** 2 层 MLP（约 155 万参数），把未见机器人的原始动作映射进共享潜在动作空间。
7. **预训练：** EgoDex（人手第一视角，138 维动作）、AgiBot（双臂人形，20 维）、DROID（单臂 Franka 夹爪，10 维）各 5 万条轨迹，共 15 万条；80k 迭代、batch 512，约 4 天、64 张 H200（前 60k 迭代 FDM 无时序注意力，后 20k 加 8 步时序）。预训练小时数原文未报告。
8. **下游本体（BFA）：** 双臂 Franka 配 Allegro 灵巧手，仅用右臂；25 维动作（16 手指关节 + 3 位置 + 6 旋转）；数据为 RoboCasa 中由 10 条遥操作示教经 MimicGen 生成的 7,265 条轨迹、22 类物体，约 20% 成功；头部相机。
9. **LAC-WM 微调（batch 256，共 60k）：** 先 IDM+FDM 以 LoRA(rank 2) 训 20k；冻结 FDM 从头训动作投影器 5k；再联合投影器与 FDM(LoRA) 35k。EAC-WM 为从头训练新的显式动作编码器 + FDM(LoRA) 的单阶段 60k。
10. **推理：** L40S 上批大小 10、预测 8 帧 256×256 约 0.1 s。

## 论文报告的结果

- **想象 rollout（Unseen Instances）：** PSNR/LPIPS/FID/FVD：EAC-WM-S 22.21/0.077/10.65/42.26；EAC-WM 24.25/0.057/9.31/30.35；LAC-WM 27.48/0.037/8.44/23.21。Unseen Categories 趋势一致（LAC-WM FVD 24.46）。
- **规划（VLA 采样 500 候选动作、世界模型打分选择，3 seeds，灵巧操作 BFA，仿真）：** 平均成功率 S.R.：VLA-mean 0.17、VLA-random 0.15、EAC-WM-S 0.18、EAC-WM 0.15、LAC-WM **0.22**；相对 EAC-WM +46.7%（相对提升，非绝对百分点），相对最强 VLA 基线 +29.0%，相对 EAC-WM-S +22.2%。分拆：Unseen Instances 0.25（EAC-WM 0.16）；Unseen Categories 0.18（EAC-WM 0.14）。
- **消融：** 去掉运动解码器与 cross-augmentation，S.R. 0.18→0.13（LAC-WM 高 38%）。
- **修改版 LIBERO（5 个含未见物体任务，各 25 rollouts，π0.5 骨干，20 个候选动作块）：** LAC-WM 0.920；EAC-WM 0.824；VLA-mean 0.816；VLA-random 0.744。11.7% 是相对提升（92.0/82.4），绝对 +9.6 点。
- **扩展性：** 预训练子集 EgoDex → +AgiBot → +DROID（各 5 万样本）；LAC-WM 成功率随本体数量上升，EAC-WM 下降（具体数值仅在图中）。固定样本数增加本体及 24k→150k 轨迹，LAC-WM 均提升；EAC-WM 只在同本体数据增多时改善。
- **附录 A.5（从训练集抽动作序列选择）：** LAC-WM S.R. 0.11 对 EAC-WM 0.09、EAC-WM-S 0.07。

## 复现与解释边界

- 论文自述局限：绝对灵巧任务成功率仍低；预训练使用运动标签，无标签视频预训练留待未来；无真机验证。
- 灵巧操作与 LIBERO 结果都是「世界模型为 VLA 候选动作打分」的规划设置，不是端到端策略成功率；基线差距在 0.02–0.07 量级，伴随 ±0.02–0.04 的 3-seed 标准差。
- 微调 GPU 数/成本、λ 权重、IDM/FDM 的 cross-attention 层数、世界模型总参数原文未报告。
- 代码链接未公开。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-lac-wm-cross-embodiment-latent-action-world-model.md
- 概念入口：wiki/methods/generative-world-models.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10846>
- arXiv HTML：<https://arxiv.org/html/2610.10846v1>
- 项目页：<https://lacwm.github.io/>

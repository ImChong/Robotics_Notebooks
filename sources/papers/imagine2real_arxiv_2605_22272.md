# Imagine2Real: Towards Zero-shot Humanoid-Object Interaction via Video Generative Priors

> 来源归档（ingest）

- **标题：** Imagine2Real: Towards Zero-shot Humanoid-Object Interaction via Video Generative Priors
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2605.22272>
- **机构：** Zhejiang University；Shanghai AI Laboratory；The Chinese University of Hong Kong
- **入库日期：** 2026-07-03
- **更新日期：** 2026-07-22
- **开源状态：** 未公开；论文称 anonymized source code and pre-trained checkpoints provided in supplementary material，但未发现公开 GitHub、项目页或可运行 README。
- **一句话说明：** 将机器人与物体运动统一为 **4D 点轨迹**，用 **BFM 潜空间** 上的稀疏关键点 Tracker（基座 + 双手 + 物体）绕过密集重定向与 CAD 几何先验；三阶段渐进训练后，在 mocap 系统内实现 zero-shot humanoid-object interaction。

## 摘要要点

- 目标：用视频生成先验解决 humanoid HOI 高保真 3D 数据稀缺问题。
- 问题 1：Representation Misalignment。机器人和物体分别估计会产生尺度、深度和坐标不一致。
- 问题 2：Retargeting Complexity。HOI 中密集人到机器人 retargeting 需要复杂 morphing，误差会被接触放大。
- 核心解：用统一 4D point trajectories 表示机器人/物体运动，只跟踪 base、hands、object 等 sparse critical points。
- 控制解：在 BFM latent space 中搜索动作，让 sparse keypoints 仍能生成自然步态和稳定全身动作。

## 方法要点

- **Video-to-motion：** 初始图像 + 文本 → Seedance 2.0 Fast 生成交互视频。
- **Point extraction：** SAM3 分割机器人双手、base 和物体；SpaTrackerV2 跟踪 mask 内点；按 visibility/confidence/outlier 过滤，几何平均为关键点轨迹。
- **BFM backbone：** 用 AMASS、LAFAN1、100STYLE 共 **68.5h** 非交互运动训练全身 motion latent。
- **Keypoints Tracker：** 在 BFM frozen predictor/decoder 上学习 latent residual，跟踪 sparse keypoints，保持自然性。
- **Interaction Adaptor：** 用 OMOMO box carry/push 子集约 **0.43h** 训练 joint residual，补足接触操作能力。
- **Mocap deployment：** mocap 提供真实机器人/物体 keypoints 与尺度标定，解决视频深度歧义。

## 实验与数字

- **训练数据：** BFM 10,000+ clips / 68.5h；Keypoints Tracker 4,000+ clips / 8.86h；Interaction Adaptor 200+ clips / 0.43h。
- **训练设置：** Isaac Gym；8192 parallel environments；PPO；单 RTX 4090；keypoints/adaptor 训练后段注入 5 cm Gaussian noise。
- **HOI 成功率：** Carry Box w/ Adaptor **82.65%**；Push Box w/ Adaptor **64.91%**。
- **Adaptor 消融：** Carry Box w/o Adaptor **0%**；Push Box w/o Adaptor **29.82%**。
- **BFM 消融：** Direct tracking 点误差最低但 action rate/smoothness 高；BFM 版本保持 **99.36%** tracking SR，同时 MPJAE、action rate、smoothness 最低。
- **真实部署：** Unitree G1 在 mocap 系统中执行 lifting boxes 和 hitting an “Iron Man” pillar 等 zero-shot 任务。

## 开源 / 复现状态

- **代码：** 未发现公开仓库；arXiv HTML 仅称匿名源码和 checkpoint 在 supplementary material。
- **数据：** 未发现公开下载入口。
- **项目页：** 未发现官方项目页链接。
- **复现边界：** 依赖外部视频模型 Seedance 2.0 Fast、SAM3、SpaTrackerV2、mocap 系统、BFM/Keypoints/Adaptor 三阶段训练配置；没有公开代码时难以复核。

## 对 wiki 的映射

- [paper-imagine2real-zero-shot-hoi](../../wiki/entities/paper-imagine2real-zero-shot-hoi.md) — 完整实体页，含流程、机制、评测与复现风险。

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2605.22272>
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

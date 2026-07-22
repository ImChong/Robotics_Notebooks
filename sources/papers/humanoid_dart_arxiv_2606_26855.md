# Humanoid-DART: Humanoid Loco-Manipulation using Diffusion-guided Augmentation through Relabeling and Tracking

> 来源归档（ingest）

- **标题：** Humanoid-DART: Humanoid Loco-Manipulation using Diffusion-guided Augmentation through Relabeling and Tracking
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.26855>
- **机构：** Max Planck Institute for Intelligent Systems 等
- **入库日期：** 2026-07-03
- **更新日期：** 2026-07-22
- **开源状态：** 待发布；论文附录写明 “We will open-source the codebase upon acceptance of the paper.”，截至 2026-07-22 未发现官方 GitHub。
- **一句话说明：** 从 2–4 条稀疏示范自举：**goal-conditioned dual-branch diffusion transformer** 生成机器人/物体轨迹，**RL motion tracker** 在物理仿真中验证并过滤 elite，配合 curriculum 与 goal relabeling 扩展 loco-manip 目标空间；在 **Unitree G1** 推、踢、handoff、pick-and-place 等技能上验证。

## 摘要要点

- 背景：人形 loco-manipulation 的连续目标空间由身体运动、物体状态、接触时序共同定义，人工示范难以覆盖。
- 目标：从少量 base demonstrations 出发，逐步扩大 goal-conditioned behavior repertoire。
- 核心：扩散生成器负责结构化轨迹探索，RL tracker 负责动态执行和物理过滤，二者交替迭代。
- 关键机制：frontier/refine/explore curriculum、goal relabeling、elite archive、dual-branch DiT、structured partial unmasking。
- 结果：pick-and-place 从 4 条 DF seed 达到 **96.4%** 目标覆盖率，并能生成比 seed 远 4–5× 的目标轨迹。

## 方法要点

- **Motion representation：** 轨迹在 robot local yaw-aligned frame 表达，包含 root displacement/height/yaw、joint positions、root pitch/roll、robot-relative object pose。
- **Generator：** dual-branch DiT1D；global stream 处理 root/object/goal，local stream 处理 body pose；local-to-global cross-attention 连接身体姿态和物体状态。
- **Inference：** 10 deterministic DDIM steps；long-horizon autoregressive stitching；history frames RePaint-style inpainting；classifier-free guidance。
- **Physics evaluator：** RL tracker rollout 后用 fitness 评估 tracking、contact、root、leg、object 误差；fall/early termination 置零。
- **Tracker：** DeepMimic-style object-contact tracker；PPO + GAE；actor 看 reference horizon/proprioception/object pose，critic 看 privileged simulation state。
- **Curriculum：** 按 elite set 到 task bins 的距离划分 refine/explore/frontier；near-miss rollout 用 achieved goal relabel。

## 实验与数字

- **任务：** push、kick、hand-off、pick-and-place。
- **初始数据：** 每任务 2–4 sparse base demonstrations。
- **迭代预算：** 4 generations × 10 iterations；每 iteration 采样 3000 candidate trajectories；每 generation 末 retrain tracker。
- **最终覆盖率：** Humanoid-DART：Push 61.1、Kick 54.8、Hand-off 51.5、P&P 96.4；Hierarchical Diff. + RL 在 P&P 仅 6.2。
- **generator ablation：** dual-branch success 1.00；single-branch success 0.25；structured unmasking 改善 path straightness 和 hand-object distance。
- **seed ablation：** P&P 中 DF 4 demos 覆盖率 96.4；KF 2 demos 覆盖率 28.8。

## 开源 / 复现状态

- **代码：** 未公开；官方承诺 upon acceptance open-source。
- **数据：** 未发现 base demonstrations / elite archive / trained models 下载。
- **项目页：** 未发现项目页。
- **复现边界：** 需要 DynaRetarget-style reference motions、mjlab/MuJoCo RL、dual-branch DiT、tracker reward/domain-randomization 配置、curriculum 参数；目前只能根据论文实现。

## 对 wiki 的映射

- [paper-humanoid-dart](../../wiki/entities/paper-humanoid-dart.md) — 完整实体页，含流程、核心机制、评测表与开源状态。

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2606.26855>
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

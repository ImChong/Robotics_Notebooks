# Zetta ζ（闭环具身 Harness · 自进化物理智能）

> 来源归档（ingest）

- **标题：** Zetta ζ: An Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.16590>
- **HF Papers：** <https://huggingface.co/papers/2608.16590>
- **机构：** 清华大学 AIR（AIR）；Z-Trans AI
- **项目页：** <https://air-embodied-brain.github.io/zetta/>
- **代码：** <https://github.com/air-embodied-brain/Zetta-Embodiment>
- **入库日期：** 2026-09-26
- **一句话说明：** 冻结基础策略/VLA，在线进化 **代码化 runtime critics + recovery skills**；三时间尺度闭环（动作频率治理、rollout 候选优化、验证门控技能更新）+ **Z-Infra**  rollout 基建；LIBERO-Pro **90.8%**、RoboCasa 18 任务 **93.6%**，相对 RPent **11.1×** 推理加速。

## 核心摘录（MVP）

### 1) 问题：开环 agent harness

- **摘录要点：** 现有具身 agent 多在 episode 结束后反思，无法在 **动作频率** 跟踪机器人—环境状态；后验反思难以复用精确失败时刻状态。
- **对 wiki 的映射：**
  - [paper-zetta](../../wiki/entities/paper-zetta.md)

### 2) 三循环自进化（不改 VLA 权重）

- **摘录要点：** Loop1 **Critic-Governed Action**（动作频率 critic 触发 recovery）；Loop2 **Rollout-Batch Candidate Optimization**（失败聚类、因果诊断、代码空间 critic/recovery 候选）；Loop3 **Validation-Gated Skill Update**（历史回归 + held-out 泛化通过才写入 skill memory）。Harness \(H=\{C,R,T\}\) 进化，策略冻结。
- **对 wiki 的映射：**
  - [paper-zetta](../../wiki/entities/paper-zetta.md)

### 3) Z-Infra 与结果

- **摘录要点：** 控制面将 agent 逻辑与异构 env/model worker 解耦；有效 rollout **1.7→35.1 ep/min（20.6×）**。LIBERO-Pro Goal 自 **34.5%→90.8%**；RoboCasa **73.6%→93.6%**；技能 **零样本迁移**（PnP-Stove→Sink/Cabinet/Toaster）；「Aha moment」式跃迁（如 wine bottle 15%→95%）。
- **对 wiki 的映射：**
  - [Zetta 项目页](../sites/zetta-air-embodied-brain.md)
  - [air-embodied-brain/Zetta-Embodiment](../repos/air-embodied-brain-zetta-embodiment.md)

## 当前提炼状态

- [x] 项目页 + GitHub 开源核查（2026-09-26）
- [x] wiki 映射：`wiki/entities/paper-zetta.md`

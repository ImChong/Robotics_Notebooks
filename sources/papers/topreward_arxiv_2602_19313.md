# TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics

> 来源归档（ingest）

- **标题：** TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics
- **类型：** paper
- **来源：** arXiv abs / PDF；项目页；官方代码仓
- **原始链接：**
  - <https://arxiv.org/abs/2602.19313>
  - <https://arxiv.org/pdf/2602.19313>
  - <https://topreward.github.io/webpage/>
  - <https://github.com/TOPReward/TOPReward>
- **作者：** Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang, Zhongzheng Ren, Lillian J. Ratliff, Jiafei Duan*, Dieter Fox*, Ranjay Krishna*（\*共同指导）
- **机构：** 华盛顿大学（UW）；艾伦人工智能研究所（AI2）；亚马逊（Amazon）；北卡罗来纳大学教堂山分校（UNC–Chapel Hill）
- **入库日期：** 2026-07-27
- **一句话说明：** 训练无关的零样本进度奖励：用预训练视频 VLM 对「任务已完成」肯定回答 token（如 `True`）的 log-likelihood，把轨迹前缀打成稠密时序进度信号；并发布真机操作奖励基准 ManiRewardBench。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 项目页 | <https://topreward.github.io/webpage/> — 含 arXiv / Code 按钮 |
| 代码 | <https://github.com/TOPReward/TOPReward> — **已开源（MIT）**；`uv` + Hydra；`predict_topreward` / `predict_gvl` |
| 基准数据 | Hugging Face：`ajyanggg/manirewardbench_*`（LeRobot / Franka / YAM 等子集）— **已公开** |
| 权重 | 不训练专用 reward 模型；依赖开源/商用视频 VLM（如 Qwen3-VL-8B、Molmo-2、Gemini） |
| 结论 | **已开源**（推理评测代码 + ManiRewardBench 数据） |

## 核心论文摘录（MVP）

### 1) 问题：通用过程奖励难迁移

- **链接：** <https://arxiv.org/abs/2602.19313>；项目页 Abstract
- **摘录要点：** VLA 预训练进步快，但真机 RL / 稠密反馈仍受稀疏奖励与低样本效率限制；已有时序价值函数跨域迁移弱，且常依赖人工进度标注、任务特定演示或在机器人数据上训 reward model。
- **对 wiki 的映射：**
  - [TOPReward（论文实体）](../../wiki/entities/paper-topreward.md)
  - [过程奖励建模](../../wiki/concepts/progress-reward-modeling.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 方法：Token 概率 → 零样本进度

- **链接：** 项目页 Method Overview；arXiv
- **摘录要点：**
  1. 指令条件下问 VLM：观测轨迹前缀是否完成该指令；
  2. 取肯定回答 token（如 `True`）的 log-likelihood，**不**依赖数值生成或强指令跟随；
  3. 沿轨迹前缀对齐，得到稠密时序进度 / 过程奖励（可 per-episode min-max 归一）。
- **对 wiki 的映射：**
  - [TOPReward](../../wiki/entities/paper-topreward.md)
  - [过程奖励建模](../../wiki/concepts/progress-reward-modeling.md) — 冻结基础模型打分范式

### 3) 评测：OXE + ManiRewardBench；对照 GVL

- **链接：** 项目页 Quantitative Results
- **摘录要点：** Open X-Embodiment（39 datasets / 780 episodes）上 Qwen3-VL-8B Mean VOC **0.857**（GVL 0.194）；自建 ManiRewardBench（113 tasks / 497 episodes，Franka / YAM / SO-100/101 等）上 Qwen3-VL Mean VOC **约 0.94–0.95**，显著优于同骨干 GVL。VOC 测秩相关，成功检测另用末帧 log-likelihood 做 ROC-AUC。
- **对 wiki 的映射：**
  - [TOPReward](../../wiki/entities/paper-topreward.md)
  - [Open X-Embodiment](../../wiki/concepts/open-x-embodiment.md)

### 4) 下游：成功检测 + TOP-AWR 真机 BC

- **链接：** 项目页 Success Detection / Real-World Deployment
- **摘录要点：** 末三帧平均 log-likelihood 做成功/失败分类；在单臂 SO-100 上用 TOPReward 作 advantage 权重（TOP-AWR），相对标准 BC 提升 6 个任务的成功次数（每任务约 50 条噪声演示）。
- **对 wiki 的映射：**
  - [TOPReward](../../wiki/entities/paper-topreward.md)
  - [AWR](../../wiki/methods/awr.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

## 关联归档

- [项目页](../sites/topreward-github-io.md)
- [代码仓](../repos/topreward.md)

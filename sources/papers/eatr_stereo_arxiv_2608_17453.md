# EATR-Stereo: Embodiment-Aware Token Routing of Paired Stereo Evidence for Humanoid Vision-Language-Action Control（arXiv:2608.17453）

> 来源归档（ingest）

- **标题：** EATR-Stereo: Embodiment-Aware Token Routing of Paired Stereo Evidence for Humanoid Vision-Language-Action Control
- **短名：** EATR-Stereo
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.17453>
- **PDF：** <https://arxiv.org/pdf/2608.17453>
- **机构：** 哈尔滨工业大学（HIT）；荣耀（Honor）
- **作者：** Songwei Wu, Rui Zhao, Fan Yang, Zhongqiang Nie, Zhiduo Jiang, Wandong Sun, Yuwei Li, Jian Hu, Yang Liu, Hong Liu
- **入库日期：** 2026-08-23
- **一句话说明：** 冻结 VLM 前提下，用 primary-aligned CVAT 与分段本体路由选择性融合头载双目证据，在 33-DoF HONOR Omega 1.0 长程任务达 60% 全流程成功。

## 开源状态（步骤 2.5，2026-08-23）

- **确认未开源**：arXiv 全文与摘要页均未列项目页、GitHub、Hugging Face 或权重链接；论文实验基于 GR00T1.7 骨干与 HONOR Omega 1.0 真机，无可公开复现入口。

## 核心摘录（面向 wiki 编译）

### 摘录 1：问题与接口设计

- 头载同步双目为人形 VLA 提供互补可见性，但现有接口常丢弃辅视图、粗暴拼接 token，或替换预训练主视图通路（如 StereoPolicy 融合表示）。
- EATR-Stereo **保留主视图 token**，用 primary-query cross-attention 从同步辅视图构造 **primary-aligned Cross-View Auxiliary Tokens（CVATs）**；**Cosmos VLM 冻结**，仅训练 CVAT、分段本体编码器、token router 与 GR00T action expert。

**对 wiki 的映射：** vla、stereo vision、proprioception、humanoid manipulation、long-horizon

### 摘录 2：分段本体路由

- **Body-segmented proprioceptive routing** 用近期 \(K\) 步 37-D 状态（33 关节 + 4-D 基座姿态四元数）按身体分段编码，对 CVAT **逐 token** 门控辅视图证据使用。
- 与 CVAT-Flat（扁平状态条件）对比：全任务 60% vs 55%，抓取 100% vs 90%，阶段 80% vs 72.5%。

**对 wiki 的映射：** whole-body control、embodiment-aware fusion

### 摘录 3：真机主结果（HONOR Omega 1.0，33-DoF，BeyondMimic WBC）

- 任务：search–approach–grasp–place–return，**>100 s**；20 trial/配置，ID/OOD 瓶位各 10。
- EATR-Stereo：**全流程 60.0%**、**抓取 100.0%**、**阶段 80.0%**（ID 90% / OOD 70%）；优于 StereoPolicy（45/85/65）与默认双图 GR00T（35/80/57.5）。
- **严重不对称遮挡恢复**：80%（8/10）vs CVAT 30%、CVAT-Flat 60%；成功 trial 平均恢复时间 **22.4 s**（CVAT-Flat 41.7 s）。
- 训练：1000 分段演示、60k step、batch 512、16×H20；相对 GR00T 仅 +5.8% 训练时长（10.95 h vs 10.35 h），阶段成功率与 Estimated-Depth VLA 同为 80% 但后者需 41.8 h。

**对 wiki 的映射：** loco-manipulation、occlusion、sim2real 真机评测

### 摘录 4：RoboCasa365 仿真（Franka，仅左右 agent 双目）

- 18 任务 × 20 trial，3600 rollout；EATR-Stereo **43.33%** 聚合成功率，领先 CVAT（39.44%）与 StereoPolicy（38.06%）。

**对 wiki 的映射：** manipulation benchmark、paired-view ablation

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-eatr-stereo.md`](../../wiki/entities/paper-eatr-stereo.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体回链

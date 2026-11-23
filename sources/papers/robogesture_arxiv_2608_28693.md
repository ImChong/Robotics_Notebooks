# RoboGesture：人形实时语义对齐伴随语音手势（arXiv:2608.28693）

> 来源归档（ingest）

- **英文标题：** RoboGesture: Real-Time Semantic-aligned Co-Speech Gestures Generation for Humanoid Interaction
- **标题：** RoboGesture：人形实时语义对齐伴随语音手势
- **类型：** paper
- **作者：** Zifan Wang, Ziang Ren, Pengyang Shi, Zirui Wang, Chenghuai Lin, Tianze Wang, Zekun Qi, Liangliang Zhao, He Wang, Li Yi（* 同等贡献；Li Yi 通讯作者）
- **机构：** Tsinghua University；Galbot Inc.；Beijing Institute of Technology；Harbin Institute of Technology；Peking University；Shanghai Qi Zhi Institute
- **arXiv：** <https://arxiv.org/abs/2608.28693>
- **PDF：** <https://arxiv.org/pdf/2608.28693>
- **项目页：** <https://RoboGesture.github.io>
- **会议：** ECCV 2026 Poster（Session 2 · ExHall #178 · 2026-09-10 16:30–18:30 CEST）
- **开源：** 已于 GitHub 发布部署推理、安全碰撞滤波、54 类 `MotionModel` / Qwen LoRA 权重及 `training_examples` 训练核心示例：<https://github.com/GalaxyGeneralRobotics/RoboGesture>（2026-11-21 复核；注：全量 1000 h / 300+ 类数据集与完整预处理/大规模训练脚本未公开）
- **核对范围：** 已通读 arXiv `2608.28693v1` 论文正文 §1–§6 与实验表（含 Table 1–6、200 次真机长时实验与延迟分解）、项目页与官方开源仓库 `@2913f1d`
- **通读：** 已完成
- **入库日期：** 2026-09-07（周更索引）；2026-09-28 补全元数据；2026-11-21 论文全文与开源仓库核对
- **策展索引：** [wechat_shenlan_weekly_papers_2026-09-04.md](../blogs/wechat_shenlan_weekly_papers_2026-09-04.md)

## 核心论文摘录

### 1) 300+ 类手势数据 + 合成管线

- MoCap→重定向→MPC 离线滤碰撞；半合成 1000 h 机器人专属音–动对。
- **对 wiki 的映射：** [../../wiki/entities/paper-robogesture.md](../../wiki/entities/paper-robogesture.md)

### 2) 分层语义–声学对齐 + DiT-CFM

- Mimi 原始音频 token；Anti-Inertia CFG Masking 防运动惯性塌缩。
- **对 wiki 的映射：** [../../wiki/entities/paper-robogesture.md](../../wiki/entities/paper-robogesture.md)

### 3) G1 + BrainCo 真机

- 41 DoF 上身；≈120 FPS 生成 + 5.6 ms MPC 安全滤波；BEAT/SemanticBEAT SOTA。
- **对 wiki 的映射：** [../../wiki/entities/paper-robogesture.md](../../wiki/entities/paper-robogesture.md)

### 4) 三大瓶颈与 robot-centric 闭环

- 论文 framing：**语义丰富数据稀缺**、**modality eclipse**（运动惯性盖过音频）、**avatar→真机安全鸿沟**；RoboGesture 在机器人表征空间 **数据–模型–控制** 联合设计，部署为 listen–respond–gesture 完整交互系统（上游 Speech-LLM + 流式 motion + 执行模块）。
- **对 wiki 的映射：** [../../wiki/entities/paper-robogesture.md](../../wiki/entities/paper-robogesture.md)

## 当前提炼状态

- [x] 公众号周更 ingest 映射
- [x] wiki 实体页
- [x] 2026-09-28 英文题名 / 作者 / 机构与项目页复核

---

type: entity
tags: [dataset, bfm, behavior-foundation-model, human-motion, awesome-bfm-papers, berkeley]
status: complete
updated: 2026-11-20
summary: "BABEL（CVPR 2021, arXiv:2106.09696）：为 AMASS 中约 43.5 小时 SMPL-H 动捕提供序列级（28k+）与精细帧级（63k+）动作语义标注（250+ 类范畴），打通动作识别、时序动作分段与语言–动作连续过渡。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
  - ./amass.md
  - ./paper-notebook-humanml3d.md
  - ./dataset-bfm-kit-ml.md
sources:
  - ../../sources/papers/bfm_awesome_dataset_babel_cvpr_2021.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
original:
  - ../../sources/papers/bfm_awesome_dataset_babel_cvpr_2021.md
---

# BABEL（AMASS 帧级与序列级动作语义数据集）

**BABEL**（*Bodies, Action and Behavior with English Labels*, CVPR 2021, [arXiv:2106.09696](https://arxiv.org/abs/2106.09696)）在统一参数化人体动捕库 [AMASS](./amass.md) 上补齐了高密度的序列级与帧级动作语义标签，是动作理解与可控运动拼接/过渡生成的关键基准。

## 一句话定义

为 AMASS 中约 43.5 小时 SMPL-H 动捕数据提供序列级（28k+）与逐帧时序对齐（63k+，涵盖 250+ 动作类别）语义标注，从而连接底层连续三维骨骼轨迹与高层语言动作标签。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| AMASS | Archive of Motion Capture as Surface Shapes | 统一转化为 SMPL/SMPL-H 网格的参数化大动捕集合 |
| SMPL-H | Skinned Multi-Person Linear model with Hands | 含手部关节参数的统一参数化人体表征 |
| MoCap | Motion Capture | 光学或惯性捕捉的高精度人体动作序列 |

## 为什么重要

- **帧级时序切分支持多动作复合过渡**：与只给整段视频打一句笼统 caption 的数据集不同，BABEL 标注出「先走、再蹲下、再举手」在每一帧的确切起止边界，直接催生 TEACH 等跨动作串联与时序条件生成方法。
- **桥接行为识别与物理技能检索**：250+ 精细动作类别与原始自然语言短语配对，让行为基础模型（BFM）能基于语义子集筛选训练或评估特定全身技能。
- **机器人技能标签入口**：[AMASS](./amass.md) 常用于人形跟踪器（如 PHC、ProtoMotions、SONIC）预训练；结合 BABEL 语义切分可直接按「转身/跳跃/蹲起」抽取测试集与技能库子集。

## 核心原理

| 字段 / 机制 | 原文核对详情 |
|-------------|--------------|
| **数据规模** | 覆盖 [AMASS](./amass.md) 约 **43.5 小时** SMPL-H 动捕，含 **28k+ 序列级标注** 与 **63k+ 帧级标注**（外部目录常见聚合统计约为 13,220 clips · 43.5 h） |
| **动作词表体系** | 从众包自由英文动词短语聚类归纳出 **250+ 动作类别**，允许多标签并发（如边走边挥手）与未完全切分的重叠时间区间 |
| **配套基准任务** | 提出序列/帧级 **Action Recognition（BABEL-60 / BABEL-120）** 及文本/动作序列合成评估协议 |
| **官方入口** | <https://babel.is.tue.mpg.de/> · [arXiv:2106.09696](https://arxiv.org/abs/2106.09696) |

## 工程实践

- **长尾类别分布处理**：BABEL 动作类别呈典型长尾（行走、站立频次远高于稀有动作），训练识别或技能检索模型时需按频率加权采样或区分常见核心类与长尾子集。
- **继承 AMASS 重定向管线**：轨迹本身为 SMPL-H 连续参数，进入人形仿真器（Isaac Gym / Isaac Lab / MuJoCo）前仍需经过脚部接地清理与 [GMR](../methods/motion-retargeting-gmr.md) 尺度重定向。

## 局限与风险

- **学术授权绑定上游**：下载 BABEL 注释需先注册并通过各个 AMASS 子数据集的非商业学术许可，不可直接未经授权用于商业闭源产品训练。
- **偏重单人无环境交互描述**：标注重点在人体自身动作动词，缺少物体位姿、接触力标签及复杂场景碰撞边界说明。

## 关联页面

- [AMASS](./amass.md) — BABEL 标注所依附的底层统一 SMPL-H 动捕库
- [KIT-ML](./dataset-bfm-kit-ml.md) — 更早期的 MMM 语言–动作句级配对数据集
- [HumanML3D](./paper-notebook-humanml3d.md) — 基于 AMASS+HumanAct12 构建的整段句级文本描述基准
- [Behavior Foundation Model](../concepts/behavior-foundation-model.md) — 行为基础模型对可索引语义运动库的依赖
- [BFM 41 篇技术地图](../overview/bfm-41-papers-technology-map.md) — BFM 数据层全景对比

## 参考来源

- [bfm_awesome_dataset_babel_cvpr_2021.md](../../sources/papers/bfm_awesome_dataset_babel_cvpr_2021.md)
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md)
- [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)
- 数据集论文与主页：<https://arxiv.org/abs/2106.09696> · <https://babel.is.tue.mpg.de/>

## 推荐继续阅读

- [awesome-bfm-papers § Datasets](https://github.com/friedrichyuan/awesome-bfm-papers#datasets)
- [BABEL: Bodies, Action and Behavior with English Labels（CVPR 2021 PDF）](https://arxiv.org/pdf/2106.09696)

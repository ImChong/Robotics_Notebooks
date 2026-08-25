# ForeTime-VLA（世界模型未来 Token 蒸馏）

> 来源归档（ingest）

- **标题：** ForeTime-VLA: Causal Future-Token Distillation from a World Action Model for Conveyor-Belt Manipulation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.20735>
- **机构：** 清华大学；上海人工智能实验室；哈尔滨工业大学；云深处科技（DEEP Robotics）
- **入库日期：** 2026-08-25
- **一句话说明：** 从冻结 Fast-WAM 系教师蒸馏 64-D 未来感知 action-equivalent 表征到因果 π₀.₅：八帧历史预测未来/阶段/过渡时间，4 个 future token + 1 个 phase token 条件 VLM，部署无需 WAM 前向。

## 核心摘录（MVP）

### 1) 动态传送带：当前观测不足以定接触时机

- **摘录要点：** 抓取移动物体需预判进入可达域、夹爪闭合与接触姿态；纯 action 回归 VLA 可隐式学到规律，但无显式未来结构监督。Fast-WAM 证明视频 建模可作训练信号、推理可去掉，ForeTime-VLA 问：能否把该结构压缩进预训练 VLA？
- **对 wiki 的映射：**
  - [ForeTime-VLA](../../wiki/entities/paper-foretime-vla.md) — 问题与教师–学生范式。
  - [π₀.₅](../../wiki/entities/paper-pi05-open-world-vla.md) — 基座策略。

### 2) 离线 action-equivalent 教师 + 因果学生

- **摘录要点：** 教师：冻结 Wan2.2 VAE 编码当前+8 未来帧（偏移 {2,4,7,9,12,14,17,19}）→ 非坍缩 adapter → 64-D 白化码 \(z^T\)。学生：八帧因果编码器预测 \(z\)、操作阶段与归一化 time-to-transition；**4 future tokens + 1 phase token** 进 VLM prefix，预测未来与过渡 horizon 条件 action expert。保留原 flow-matching action 目标，叠加 cosine / Gram 几何 / phase / TTT / action-equivalence 损失。
- **对 wiki 的映射：**
  - [ForeTime-VLA](../../wiki/entities/paper-foretime-vla.md) — 架构与损失。
  - [World Action Models](../../wiki/concepts/world-action-models.md) — WAM 训练期特权、部署期因果。

### 3) 离线窗与真机传送带

- **摘录要点：** 去重传送带集 768 匹配窗/划分：test MAE **0.134119→0.130593**（−2.63%）、L2 −3.02%，延迟 +2.46–2.93%。真机：静止 **81.1%**、慢速 **58.9%** grasp SR（次优 +12.2 / +22.2 pt）；三档带速 **44/90** vs π₀.₅ **23/90**（快速 **11/30 vs 2/30**）。
- **对 wiki 的映射：**
  - [ForeTime-VLA](../../wiki/entities/paper-foretime-vla.md) — 指标读法。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 动态抓取语境。

### 4) 开源状态（截至 2026-08-25）

- **摘录要点：** 用户未提供项目页/GitHub；arXiv 正文与机构信息为准 → **截至入库日未列官方代码或权重 URL**。
- **对 wiki 的映射：**
  - [ForeTime-VLA](../../wiki/entities/paper-foretime-vla.md) — 局限节标明待发布。

## 当前提炼状态

- [x] arXiv 摘要与正文方法节已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-foretime-vla.md` 新建

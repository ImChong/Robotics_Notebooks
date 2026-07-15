# Xiaomi-Robotics-U0

> 来源归档（ingest）

- **标题：** Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model
- **类型：** repo + paper（arXiv）+ 官方项目页
- **组织：** Xiaomi Robotics（小米机器人实验室）
- **代码：** https://github.com/XiaomiRobotics/Xiaomi-Robotics-U0
- **品牌站说明页：** https://robotics.xiaomi.com/xiaomi-robotics-u0.html
- **论文：** https://arxiv.org/abs/2607.11643
- **论文 HTML：** https://arxiv.org/html/2607.11643v1
- **入库日期：** 2026-07-15
- **一句话说明：** **38B** 多模态 **自回归世界基础模型**：在 **EMU3.5（Qwen3-32B + IBQ 图像 tokenizer）** 上持续训练，用统一 **next-token prediction** 联合 **T2I / X2I / 多视角具身场景生成 / 具身迁移 / 具身视频**；**FlashAR+** + **vLLM** 将 1024² 单图延迟从 **~451s 降至 ~5.4s**；**WorldArena #1**、人类评测优于 **GPT-Image-2** 的多视角场景与迁移；**π₀.₅** 真机 OOD 干扰场景完成度 **36.9%→63.2%**（风格迁移增广）。
- **沉淀到 wiki：** [Xiaomi-Robotics-U0](../../wiki/entities/xiaomi-robotics-u0.md)

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Generative World Models](../../wiki/methods/generative-world-models.md) | **世界基础模型 → 具身合成** 的统一自回归范式；保留互联网 T2I/X2I 能力同时做多视角机器人观测 |
| [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md) | 同实验室 **VLA（4.7B）** 与 **WM（38B）** 互补：U0 作 **可扩展数据引擎** 提升下游策略 OOD 鲁棒性 |
| [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) | 静态场景合成 → 时序 rollout → 闭环 **agentic 数据生成** |
| [EWMBench](../../wiki/entities/ewmbench.md) | 具身视频质量评测语境；U0 在 **WorldArena** 报告 SOTA |
| [Manipulation](../../wiki/tasks/manipulation.md) | 真机三任务（耳塞入盒 / 折毛巾 / 装箱）与 **进度里程碑** 评测 |

---

## 设计要点（官网 / 论文归纳）

1. **统一 formulation：** 多模态上下文 \(\mathcal{C}\) 与输出序列 \(\mathcal{Y}\) 均为 **文本 / 图像 / 机器人控制 token** 的离散序列，全部用 **NTP** 优化。
2. **单步任务：** T2I、X2I（1–3 张参考图 + 文本）、**Embodied Scene Generation**（本体 + 场景描述 → 多视角初始观测）、**Embodied Transfer**（当前多视角观测 + 目标场景 → 迁移后多视角 RGB）。
3. **序列任务：** **子任务–子目标图文交错**；**1/3/5 FPS** 多帧率操纵视频，支持稀疏规划与稠密动力学。
4. **架构：** 初始化 **EMU3.5**；**IBQ** tokenizer（16×16 空间压缩）；无任务专用头，全部 **单序列 AR**。
5. **数据（规模）：** 单步 **9.5M 样本 / 56.4B tokens**；序列 **2.6M 视频 / 49.6B tokens**；六域（通用图文、具身操纵、自动驾驶、自我中心、3D 重建、游戏）+ **Qwen3-VL-235B** 统一标注管线（五维结构化场景：workspace / background / foreground irrelevant / target objects / lighting）。
6. **训练分两阶段：** **Single-step** 四任务共训防遗忘；**Sequential** 子任务交错 + 多 FPS 视频。
7. **FlashAR+：** 目标图像区 **反对角并行解码** + 前缀条件 **step-causal mask**；配合 **vLLM** paged KV；H20 上 1024² **450.77s → 5.44s**（官网 **82.9×** 叙事含 AR 基线对比）。
8. **下游：** 结构化 **embodied transfer** 对四任务示教做 **零样本场景增广**；**π₀.₅** 后训练，干扰组（换桌布 / 光照）平均完成度 **+26.3 pts**。

---

## 对 wiki 的映射

- 新建 **`wiki/entities/xiaomi-robotics-u0.md`**：38B 统一具身合成实体页（训练数据流 + 能力四象限 + 策略增广 Mermaid）。
- 更新 `wiki/entities/xiaomi-robotics-0.md`、`wiki/methods/generative-world-models.md`：小米 **VLA + WM** 谱系互链。

---

## 外部参考（便于复核）

- Li et al., *Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model*, arXiv:2607.11643
- [XiaomiRobotics/Xiaomi-Robotics-U0（GitHub）](https://github.com/XiaomiRobotics/Xiaomi-Robotics-U0)
- [Robotics @ Xiaomi 说明页](https://robotics.xiaomi.com/xiaomi-robotics-u0.html)

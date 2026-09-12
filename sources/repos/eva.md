# EVA（haohq19/eva）

> 来源归档

- **标题：** Maximizing Asynchronicity in Event-based Neural Networks
- **类型：** repo / event-camera / a2s / linear-attention / feature-learning
- **来源：** 清华大学 · 苏黎世大学 RPG（Haiqing Hao、Nikola Zubić、Davide Scaramuzza、Wenhui Wang 等）
- **链接：** <https://github.com/haohq19/eva>
- **论文：** <https://arxiv.org/abs/2505.11165>（ICLR 2026）
- **后续：** [SSLA-Det / haohq19/ssla](https://github.com/haohq19/ssla) — 见 [`ssla.md`](ssla.md)
- **入库日期：** 2026-09-12
- **一句话说明：** **EVA**（EVent Asynchronous feature learning）：A2S 框架，把 NLP 线性注意力与自监督思路用于 **逐事件特征**；识别任务超 prior A2S，并 **首个** 在 Gen1 检测达 **0.477 mAP** 的 A2S 方法。
- **沉淀到 wiki：** 交叉引用 [`wiki/entities/paper-sa-2603-06228-low-latency-event-based-object-detection-with.md`](../../wiki/entities/paper-sa-2603-06228-low-latency-event-based-object-detection-with.md)（SSLA 前置对照）

---

## 核心定位

ICLR 2026 *Maximizing Asynchronicity in Event-based Neural Networks* 官方代码：**Asynchronous-to-Synchronous（A2S）** 范式下的事件特征学习，为后续 **SSLA-Det** 的线性注意力 + 检测栈提供直接前驱。

---

## 仓库入口

| 组件 | 说明 |
|------|------|
| 安装 | conda；PyTorch 2.5.1 + cu124；tensorboard；tqdm |
| 数据（示例） | DVS128 Gesture：Box 下载 → `preprocess_data/preprocess_dvs128gesture.py` |
| 训练 | `python run_training.py`（`configs/train.yaml`） |
| 表征导出 | `python hidden.py --checkpoint <path>`（`configs/hidden.yaml`） |
| 模型 | RWKV6 / ResNet head / event embedding（`models/`） |

---

## 与 SSLA 的关系

| 轴 | EVA | SSLA-Det |
|----|-----|----------|
| 任务 | A2S **特征**（识别 + 首次 Gen1 检测 0.477 mAP） | 端到端 **SSLA 检测器**（Gen1 0.375 / N-Caltech101 0.515 mAP，>20× ↓ per-event 计算） |
| 注意力 | 线性注意力 + 自监督（语言建模类比） | **Spatially-sparse** 线性注意力 + MOS 状态分解 |
| 代码仓 | `haohq19/eva` | `haohq19/ssla` |

选型：若瓶颈是 **per-event FLOPs / 延迟**，优先评估 SSLA-Det；若需要 **更高 Gen1 mAP 的 A2S 特征基线**，对照 EVA hidden 表征 + 下游头。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [ssla.md](ssla.md) | 同团队后续检测专用仓 |
| [paper-sa-2603-06228](../../wiki/entities/paper-sa-2603-06228-low-latency-event-based-object-detection-with.md) | SSLA 论文实体 |

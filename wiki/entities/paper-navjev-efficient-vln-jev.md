---
type: entity
tags: [paper, vln, vision-language-navigation, system-one-model, jev, zero-shot, tongji]
status: complete
updated: 2026-09-30
arxiv: "2609.34969"
related:
  - ./typesafe-jev.md
  - ./laya.md
  - ./valen.md
  - ../tasks/vision-language-navigation.md
  - ../overview/vln-open-source-repro-paradigms.md
  - ./dimensional-can-jev-nav-benchmark.md
  - ./dimensionalos-dimos.md
  - ../comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md
sources:
  - ../../sources/papers/navjev_arxiv_2609_34969.md
  - ../../sources/sites/navjev_project.md
summary: "NavJev（arXiv:2609.34969，同济）：VLN-CE 零样本框架以 ACVC（waypoint+BLIP+RAM）与 DASM 构造紧凑动作文本 state，Jev 做 typed waypoint/STOP 选择；R2R-CE 27.0% SR / 22.4% SPL、0.65 s/步，真机办公室+咖啡厅 SR 50%；项目页未列代码仓。"
---

# NavJev：ACVC + DASM + Jev 的高效 VLN-CE

**NavJev**（*Efficient Vision-Language Navigation via Action-Centric Visual Compression and Discriminative Action-Semantic Memory*，[arXiv:2609.34969](https://arxiv.org/abs/2609.34969)，[项目页](https://kai-sheng-caesar.github.io/NavJev/)）由 **同济大学** 提出：把 **逐步 MLLM 自回归推理** 换成 **「视觉 → 动作中心文本 → Jev typed 选择」** 两阶段，使 **零样本 VLN-CE** 在 **R2R-CE** 上达到 **可比的 SR/SPL**，同时 **步级延迟与 GPU 占用** 接近轻量 VLM 策略而非 32B 级 MLLM。

## 一句话定义

**每个导航步只问 Jev「选哪个 waypoint 或 STOP」——先用 ACVC/DASM 把全景 RGB-D 压成带判别语义的选项列表，再用 System One 一次出概率，而不是每步跑一整段 MLLM 生成。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 按自然语言指令在环境中导航 |
| VLN-CE | VLN in Continuous Environments | 连续空间 waypoint 导航（无离散图） |
| ACVC | Action-Centric Visual Compression | 动作中心视觉压缩（几何+BLIP+RAM） |
| DASM | Discriminative Action-Semantic Memory | 判别性动作语义记忆 |
| SR | Success Rate | 成功到达并 STOP 的比例 |
| SPL | Success weighted by Path Length | 路径效率加权成功率 |
| R2R-CE | Room-to-Room Continuous Environment | 常用 VLN-CE 基准 |

## 为什么重要

- **对齐 VLN 决策结构：** 每步动作空间本是 **有限 waypoint + STOP**；NavJev 与 [Jev](./typesafe-jev.md) 的 **Choice** 接口一致，避免 **「为写理由而生成整段文本」** 的算力浪费。
- **效率–精度 Pareto：** 项目页报 **0.65 s/步、4.46 GiB 峰值** vs **P2DNav（Qwen3-VL-32B）4.92 s/步**；SR **27.0%** 低于 P2DNav **50%**，但 **显著快于** StreamVLN / JanusVLN 等 **7B 级** 方法（Table 2）。
- **与 Dimensional「Can Jev Nav?」互证：** [Nav Arena](./dimensional-can-jev-nav-benchmark.md) 显示 **裸 WorldState + Jev** 在长距 object-goal 上弱于 **规划工具**；NavJev 回答 **「若把视觉 diligently 压成 Jev 友好文本，VLN 指令跟随能否划算？」** — 真机小样本 **SR 50%** 支持 **typed 路线在物理环境可迁移**。
- **记忆设计可复用：** DASM 的 **共享 tag 过滤 + 动作专属证据** 对任何 **「多候选、语义重叠」** 的离散控制（抓取姿态、技能 ID）有参考价值。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 同济大学（Tongji University） |
| **arXiv** | [2609.34969](https://arxiv.org/abs/2609.34969) |
| **项目页** | <https://kai-sheng-caesar.github.io/NavJev/> |
| **开源** | **截至 2026-09-30 项目页未列 GitHub/权重** — 复现待官方发布 |

## 核心原理

| 模块 | 作用 |
|------|------|
| **Waypoint 候选** | 由 VLN-CE 标准管线从 12 视角 RGB-D 生成 **M_t** 个可导航点 + **Stop** |
| **ACVC** | 每候选：**相对方向/距离** + **BLIP** caption + **RAM** 语义 tag → 紧凑文本动作描述 |
| **DASM** | 去掉多候选 **共享** tag；保留 **选中动作** 与 **历史判别** 语义，跨步累积 |
| **Jev** | 读指令 + 压缩 state + memory → **typed option**（waypoint id 或 STOP） |

### 流程总览

```mermaid
flowchart LR
  PANo["全景 RGB-D\n12 views"]
  WP["Waypoint 候选集"]
  ACVC["ACVC\nBLIP + RAM + 几何"]
  DASM["DASM\n滤共享 · 留判别"]
  JEV["Jev typed 选择"]
  ACT["执行 waypoint / STOP"]
  PANo --> WP --> ACVC --> DASM --> JEV --> ACT
  ACT --> PANo
```

## 实验与评测

### R2R-CE（项目页 Table 1–3，零样本）

| 方法 | SR ↑ | SPL ↑ | 步时 (s) ↓ |
|------|------|-------|------------|
| P2DNav (Qwen3-VL-32B) | 50.0 | 30.6 | 4.92 |
| SmartWay (GPT-4o) | 29.0 | 22.5 | — |
| Open-Nav (GPT-4o) | 19.0 | 16.1 | — |
| **NavJev (Jev)** | **27.0** | **22.4** | **0.65** |

- **决策模型对照（同 ACVC/DASM，Table 3）：** Jev **27.0% SR** vs GPT-4o **14.0%** vs Qwen3.8-Max **25.0%**；Jev **决策延迟 ~0.53 s**，**100 episode 成本 ~$0.06**。

### 消融（Table 4）

- **BLIP+RAM 无 DASM：** SR **21.0%** → 全开 **27.0%**，SPL **16.1 → 22.4** — DASM 对 **路径质量** 关键。

### 真机（办公室 + 咖啡厅，各 10 任务）

| 决策模型 | 平均 OSR | 平均 SR | 决策延迟 |
|----------|----------|---------|----------|
| Qwen3.8-Max | 35.0% | 35.0% | 1.10 s |
| **NavJev** | **60.0%** | **50.0%** | **0.65 s** |

## 源码运行时序图

**不适用**（截至入库日 **无官方可运行代码仓库**；Jev 为 **TypeSafe 托管 API**，ACVC 依赖 BLIP/RAM 等外部模型 — 待作者发布实现后再补 sequenceDiagram）。

## 工程实践

| 项 | 建议 |
|----|------|
| **API** | 需 **TYPESAFE_API_KEY**（见 [Jev 实体](./typesafe-jev.md)） |
| **感知栈** | ACVC 假设 **BLIP + RAM** 在线；部署时预算 **caption/tag 延迟** 是否仍 &lt; MLLM 单步 |
| **与规划栈组合** | 长距/低层避障仍可能需要 **Nav2 / DimOS A\***（见 [Can Jev Nav?](./dimensional-can-jev-nav-benchmark.md)） |
| **state 语言** | 与 Nav Arena 一致：优先 **robot-frame + 离散语义 helper**，勿裸世界坐标 |

## 局限与风险

- **精度上限：** SR **27%** 仍 **低于** 监督 SOTA（NavFoM **61.7%** 等）与 **P2DNav 零样本 50%** — 适合 **延迟/成本敏感** 而非刷榜。
- **细粒度空间关系：** 论文 §5.6：指令含 **「楼梯左侧第二走廊」** 类关系时，**视觉语义相近的 waypoint** 仍易混淆。
- **代码未开源：** 无法审计 ACVC/DASM 实现细节与 R2R-CE 复现脚本。
- **Jev 闭源权重：** 与 [Laya](./laya.md)/[Valen](./valen.md) 不同，**不能** 本地替换 Jev 做 ablation。

## 结论

**NavJev 表明 VLN-CE 零样本可在「每步 typed 选 waypoint」范式下同时拿到可用 SPL 与亚秒级步时，但 SPL 仍明显低于大型 MLLM 方法；工程上必须把视觉压成 Jev 友好文本（ACVC/DASM），并预期长距导航仍需规划或工具层补位。**

1. **真影响：决策范式** — 用 **有限动作 typed 选择** 替代 **逐步 MLLM 生成**，步时 **~0.65 s**、显存 **~4.5 GiB** 量级。
2. **真影响：DASM** — 对 **SPL（22.4 vs 16.1）** 提升大于仅加 RAM tag，说明 **判别性记忆** 比堆更多 caption 更重要。
3. **真影响：成本** — **~$0.06 / 100 episodes** 级决策成本，适合 **高频 closed-loop** 原型。
4. **次要代价：SR** — **27% SR** 不适合 alone 作为产品导航核心；与 **Dimcode/Nav2** 分层更现实。
5. **部署读法：** 真机 **50% SR** 小样本积极，但 **场景仅 2 处**；仿真到真机 **感知标注** 仍是瓶颈（与 [Can Jev Nav?](./dimensional-can-jev-nav-benchmark.md) 真机段落一致）。
6. **开源：** 跟踪项目页是否发布 **代码 + ACVC 脚本**；当前仅论文与网页结果可引用。

## 关联页面

- [Jev（TypeSafe System One）](./typesafe-jev.md) — 决策 API 与 SDK
- [Can Jev Nav?（Nav Arena）](./dimensional-can-jev-nav-benchmark.md) — 同 Jev、不同任务（object-goal vs 语言指令）
- [视觉–语言导航任务](../tasks/vision-language-navigation.md) — VLN-CE 问题定义
- [DimOS](./dimensionalos-dimos.md) — Dimensional 导航/sim 栈与 eval 套件
- [VLN 开源复现范式概览](../overview/vln-open-source-repro-paradigms.md)

## 参考来源

- [NavJev arXiv 归档](../../sources/papers/navjev_arxiv_2609_34969.md)
- [NavJev 项目页归档](../../sources/sites/navjev_project.md)
- [arXiv:2609.34969](https://arxiv.org/abs/2609.34969)

## 推荐继续阅读

- [NavJev 项目页（Table / 视频）](https://kai-sheng-caesar.github.io/NavJev/)
- [Jev 官方介绍](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- P2DNav（同组 prior）：[arXiv:2605.19634](https://arxiv.org/abs/2605.19634)

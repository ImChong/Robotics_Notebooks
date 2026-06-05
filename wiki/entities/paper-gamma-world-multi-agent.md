---
type: entity
tags:
  - paper
  - world-models
  - video-generation
  - multi-agent
  - interactive-simulation
  - diffusion
  - nvidia
status: complete
updated: 2026-05-30
arxiv: "2605.28816"
code: https://github.com/nv-tlabs/Gamma-World
related:
  - ../methods/generative-world-models.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../concepts/video-as-simulation.md
  - ../methods/marl.md
  - ../entities/paper-wem-world-ego-modeling.md
  - ../entities/nvidia-omniverse.md
sources:
  - ../../sources/papers/gamma_world_arxiv_2605_28816.md
  - ../../sources/sites/nvidia-gamma-world-project.md
  - ../../sources/repos/gamma_world.md
summary: "Gamma-World（arXiv:2605.28816）：多智能体生成式交互世界模型，SRAE 置换对称身份 + Sparse Hub 线性跨体注意力 + 块因果学生 24 FPS；2 人训练零样本泛化 4 人，并展示真实多机协调。"
---

# Gamma-World（γ-World / Generative Multi-Agent World Model）

**Gamma-World**（*γ-World*，arXiv:2605.28816，[项目页](https://research.nvidia.com/labs/sil/projects/gamma-world/)，[代码](https://github.com/nv-tlabs/Gamma-World)，[HF Papers](https://huggingface.co/papers/2605.28816)）提出面向 **超过两个可控主体** 的 **生成式多智能体世界模型**：在共享、持续演化的环境中，根据 **各智能体独立动作流** 交互生成一致的未来帧，并强调 **置换对称身份编码**、**可扩展跨体通信** 与 **实时动作响应 rollout**。

## 一句话定义

**用正单纯形旋转相位区分多智能体且保持置换等价，以 hub 注意力线性扩展跨体交互，再蒸馏为块因果学生，在多人虚拟与真实多机场景中实现高保真、动作可控的共享世界 rollout。**

## 为什么重要

- **补齐「多人同世界」缺口：** 单流世界模型难以表达 **多玩家 / 多机器人同时行动** 时的身份、视角与交互一致性。
- **身份编码可扩展：** 避免 **固定 slot ID** 与 **学习排序** 带来的组合爆炸；**2→4 玩家零样本** 说明编码与注意力设计服务于 **主体数扩展**。
- **工程上逼近控制环：** **24 FPS** + **KV cache** 流式生成，把「视频世界模型」从离线预览推向 **可交互模拟器** 候选（与 [Video-as-Simulation](../concepts/video-as-simulation.md) 动机一致）。
- **机器人语境：** Gallery 含 **真实多机器人协调**；与 [MARL](../methods/marl.md) 的「共享环境多体决策」互补——本文提供 **像素级可 roll 的生成环境**，而非直接给出 RL 算法。

## 核心结构

| 模块 | 作用 |
|------|------|
| **Simplex Rotary Agent Encoding（SRAE）** | **3D RoPE** 的无参数扩展：各 agent 对应旋转角空间 **正单纯形顶点** → **不同相位** + **置换等价**；无需 per-slot 可学习 ID |
| **Sparse Hub Attention** | 可学习 **hub token** 汇聚跨 agent 信息；跨体注意力 **\(O(N)\)** 而非全连接 **\(O(N^2)\)** |
| **双向多智能体教师** | 全上下文 **扩散教师**，建模完整时空与跨体依赖 |
| **块因果学生 + 蒸馏** | **块级因果** 自回归 + **KV cache**；动作响应 **24 FPS** 流式 rollout |
| **输入/输出** | 输入：**per-agent action streams**；输出：**共享、多视角** 的未来视频 rollout |

### 流程总览

```mermaid
flowchart TB
  subgraph in [多智能体输入]
    A1[Agent 1 动作流]
    A2[Agent 2 动作流]
    AN[Agent N 动作流]
    O0[当前多视角观测]
  end
  subgraph enc [身份与时空编码]
    SRAE[Simplex Rotary Agent Encoding\n正单纯形顶点相位 · 置换对称]
    ROPE[3D RoPE 时空位置]
  end
  subgraph attn [跨体交互]
    HUB[Sparse Hub Attention\nhub 中介 · O(N) 跨体代价]
  end
  subgraph train [训练]
    TCH[双向多智能体扩散教师\n全上下文]
    STU[块因果学生]
    TCH -->|蒸馏| STU
  end
  subgraph out [推理]
    KV[KV cache 流式块生成]
    VID[共享世界多视角视频\n~24 FPS · 动作响应]
  end
  A1 --> SRAE
  A2 --> SRAE
  AN --> SRAE
  O0 --> ROPE
  SRAE --> HUB
  ROPE --> HUB
  HUB --> TCH
  STU --> KV --> VID
```

## 方法栈

- **生成骨干：** 交互式 **视频扩散** 世界模型（教师全上下文；学生块因果）。
- **多智能体归纳偏置：** **几何相位**（单纯形）替代 **离散 slot embedding**；**hub** 替代 **dense all-to-all**。
- **实时化路径：** 教师–学生 **蒸馏** + **因果块** + **KV cache**（与 Matrix-Game 3.0 等「流式交互 WM」同赛道，见 sources 论文摘录中的相邻工作列表）。

## 实验与评测

- **设定（项目页 / 摘要）：** 多人 **虚拟游戏** 环境；对比 **slot-based** 与 **dense cross-agent attention** 基线。
- **报告维度：** **视频保真**、**动作可控性**、**智能体间一致性**（inter-agent consistency）。
- **泛化：** **2 玩家训练 → 4 玩家测试**，**无额外训练**（依赖 SRAE 置换对称）。
- **定性扩展：** **Real-World Robotics Coordination**（多机协调场景）；定量闭环机器人成功率 **未在公开摘要中作为主指标**。

## 与其他工作对比

| 方向 | 代表 | 与 γ-World 的差异 |
|------|------|-------------------|
| 单智能体视频 WM | UniSim、Cosmos 类 | 单动作流；无跨体身份与 hub 通信 |
| World–Ego 解耦 | [WEM](./paper-wem-world-ego-modeling.md) | 长程 **导航–操作** 混合与 world/ego 分解；非多玩家对称编码 |
| 多智能体 RL | [MARL](../methods/marl.md) | 学策略而非生成 **可控像素未来** |
| 相邻 2026 预印本 | MultiWorld、ActionParty | 同赛道「多体 / 多主体绑定」；本库尚未单独 ingest |

## 常见误区与局限

- **误区：** 把 **24 FPS 生成** 等同于 **可用于 RL 的物理正确模拟器**；仍需检验 **动作–结果对齐** 与 **闭环任务增益**（见 [训练闭环 taxonomy](../overview/robot-world-models-training-loop-taxonomy.md)）。
- **局限：** 公开材料以 **虚拟多人 + 部分真实多机 demo** 为主；**接触力级精度**、**操纵评测协议** 与 **与 Isaac/MuJoCo 混合管线** 的对接需读全文与代码。

## 关联页面

- [Generative World Models](../methods/generative-world-models.md) — 生成式世界模型总览与工程折中
- [机器人世界模型：训练闭环与三线 taxonomy](../overview/robot-world-models-training-loop-taxonomy.md) — ③ 可控视频生成 · 多体案例
- [Video-as-Simulation](../concepts/video-as-simulation.md) — 用生成交互视频作中间环境
- [MARL](../methods/marl.md) — 多机器人学习语境
- [WEM（World-Ego Model）](./paper-wem-world-ego-modeling.md) — 单主体长程 world/ego 解耦对照

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| WM | World Model | 学习环境动态以供想象/规划的世界模型 |
| Ego | Egocentric Vision | 第一人称视角感知与控制 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |

## 参考来源

- [Gamma-World 论文归档（arXiv:2605.28816）](../../sources/papers/gamma_world_arxiv_2605_28816.md)
- [NVIDIA SIL 项目页](../../sources/sites/nvidia-gamma-world-project.md)
- [nv-tlabs/Gamma-World 代码索引](../../sources/repos/gamma_world.md)

## 推荐继续阅读

- [arXiv 摘要与 PDF](https://arxiv.org/abs/2605.28816)
- [项目页 Gallery](https://research.nvidia.com/labs/sil/projects/gamma-world/) — 双人 / 四人 / 真实多机定性
- [GitHub 仓库](https://github.com/nv-tlabs/Gamma-World) — 实现与复现入口

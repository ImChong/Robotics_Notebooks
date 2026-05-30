# Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players（arXiv:2605.28816）

> 来源归档（ingest）

- **标题：** Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players
- **缩写：** **γ-World** / **Gamma-World**
- **类型：** paper / 交互式视频世界模型 / 多智能体生成
- **arXiv：** <https://arxiv.org/abs/2605.28816>（PDF：<https://arxiv.org/pdf/2605.28816>）
- **项目页：** <https://research.nvidia.com/labs/sil/projects/gamma-world/>
- **代码：** <https://github.com/nv-tlabs/Gamma-World>
- **Hugging Face Papers：** <https://huggingface.co/papers/2605.28816>
- **机构：** NVIDIA、清华大学、多伦多大学、Vector Institute 等（*Equal contribution；†Joint advising）
- **入库日期：** 2026-05-30
- **一句话说明：** 面向 **多人/多机共享环境** 的生成式交互世界模型：用 **Simplex Rotary Agent Encoding** 实现无学习槽位的置换对称智能体身份，用 **Sparse Hub Attention** 将跨智能体注意力从 \(O(N^2)\) 降到线性，并经 **双向教师 → 块因果学生蒸馏** 达到 **24 FPS** 动作响应 rollout；在虚拟多人环境上优于 slot / 稠密注意力基线，且 **2→4 玩家零样本泛化**（无需额外训练）。

## 摘要级要点

- **问题：** 交互式视频世界模型多聚焦 **单智能体**（单动作流、单可控视角）；而游戏、协作操纵、多机器人编队等场景要求 **多个主体在同一演化世界中同时可控**。
- **设计目标：** 各智能体 **独立可控**、**置换对称**（不依赖固定 agent 顺序或 per-slot 可学习 ID）、**高效推理**，并在时间与多视角上保持 **共享世界一致性**。
- **Simplex Rotary Agent Encoding（SRAE）：** 在 **3D RoPE** 上作 **无参数** 扩展：把各智能体映射为旋转角空间中 **正单纯形顶点**，赋予 **不同相位** 且全体 **置换等价**；无需 learned per-slot identity 或固定排序。
- **Sparse Hub Attention：** 可学习 **hub token** 中介跨智能体通信，跨智能体注意力代价由 **二次** 降为 **对智能体数线性**。
- **实时 rollout：** **双向多智能体教师** 指导 **块因果学生** 蒸馏；最终因果模型可用 **KV cache** 流式生成，实现 **24 FPS**、动作响应式 rollout。
- **实验（项目页 / HF 摘要）：** 多人虚拟环境上在 **视频保真、动作可控性、智能体间一致性** 上优于 slot-based 与 dense-attention 基线；**2 人训练可泛化到 4 人**；Gallery 展示 **真实多机器人协调** 扩展（超越纯虚拟游戏）。

## 核心论文摘录（MVP）

### 1) 从单智能体到「共享世界中的多主体」

- **链接：** <https://arxiv.org/abs/2605.28816> Abstract；<https://research.nvidia.com/labs/sil/projects/gamma-world/>
- **摘录要点：** 未来观测由 **多条独立动作流** 共同驱动；难点是 **跨智能体一致的世界演化** 与 **各主体可区分、可重排的身份编码**。
- **对 wiki 的映射：**
  - [Gamma-World（多智能体生成式世界模型）](../../wiki/entities/paper-gamma-world-multi-agent.md) — 问题与总体架构。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — ③ 可控视频生成支路中的 **多主体** 样本。

### 2) SRAE + Sparse Hub Attention

- **链接：** 项目页 Method 区；论文方法节
- **摘录要点：** SRAE 用 **正单纯形顶点相位** 区分 agent 且保持置换等价；Sparse Hub 用 hub 汇聚跨 agent 信息，复杂度随 \(N\) 线性扩展。
- **对 wiki 的映射：**
  - [Gamma-World](../../wiki/entities/paper-gamma-world-multi-agent.md) — 架构表与 mermaid 流程。

### 3) 教师–学生蒸馏与 24 FPS

- **摘录要点：** 全上下文 **扩散教师** 蒸馏为 **块因果学生**，块级自回归 + KV cache → **实时**、**动作条件** 交互。
- **对 wiki 的映射：**
  - [机器人世界模型训练闭环 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) — 「动作可控 + 训练有用」门槛下的 **多机 rollout** 案例。
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 多人/多机「想象环境」用于策略或规划的前置模拟器。

### 4) 2→4 玩家零样本与真实机器人 Gallery

- **摘录要点：** 置换对称编码使 **智能体数扩展** 不必重训 slot；项目 Gallery 含 **Real-World Robotics Coordination** 定性。
- **对 wiki 的映射：**
  - [MARL](../../wiki/methods/marl.md) — 多机器人 **协调** 与 **共享环境动力学** 的学习语境（本文偏生成式模拟而非 RL 算法）。
  - [Gamma-World](../../wiki/entities/paper-gamma-world-multi-agent.md) — 局限：仍以 **视频交互生成** 为主，精确接触力与闭环任务增益需单独验证。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-gamma-world-multi-agent.md`](../../wiki/entities/paper-gamma-world-multi-agent.md)
- 项目页归档：[`sources/sites/nvidia-gamma-world-project.md`](../sites/nvidia-gamma-world-project.md)
- 代码索引：[`sources/repos/gamma_world.md`](../repos/gamma_world.md)

## 相邻工作（HF Librarian Bot 推荐，未单独 ingest）

- MultiWorld（arXiv:2604.18564）— 多智能体多视角视频世界模型
- ActionParty（arXiv:2604.02330）— 生成式游戏中多主体动作绑定
- Matrix-Game 3.0（arXiv:2604.08995）— 实时流式交互世界模型 + 长程记忆

## 当前提炼状态

- [x] 摘要、三大机制与实时蒸馏叙事
- [x] 项目页 / arXiv / GitHub / HF 入口
- [ ] 细读 PDF 后补充定量表、训练数据与基线命名

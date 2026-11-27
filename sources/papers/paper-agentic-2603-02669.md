# IMR-LLM: Industrial Multi-Robot Task Planning and Program Generation using Large Language Models

- **作者：** Xiangyu Su, Juzhan Xu, Oliver van Kaick, Kai Xu, Ruizhen Hu
- **机构：** Shenzhen University、SpeedBot Robotics Co., Ltd.、Carleton University、Institute of AI for Industries, Chinese Academy of Sciences
- **版本 / 载体：** arXiv:2603.02669v1 [cs.RO]（2026-03-03，arXiv HTML 全文）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2603.02669) · [arXiv HTML 全文](https://arxiv.org/html/2603.02669)
- **项目页 / 代码：** 未确认开源代码（arXiv 摘要页、HTML 正文及索引材料均未给出独立项目页或官方代码仓库链接；文中所提 KunWu 仿真平台指向 SpeedBot 官网参考条目 [34]，非 IMR-LLM 开源代码仓库）
- **核对范围：** 通读全文 I Introduction、II Related Work、III Definitions（III-A 扩展析取图、III-B 问题形式化）、IV Method（IV-A 任务规划、IV-B 程序生成、IV-C 执行）、V Experiments（V-A IMR-Bench 与五项指标、V-B 仿真主实验 Table I 与消融 Table II、V-C 三臂真机搬运实验）及 VI Conclusion；无官方代码仓库可供静态代码核对。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[IMR-LLM 论文实体页](../../wiki/entities/paper-agentic-2603-02669.md)

### 1. 问题定位与形式化

论文针对工业产线中的**机器人作业车间调度问题（RJSSP）**与底层可执行代码生成。与经典 JSSP 相比，RJSSP 不仅对加工设备（如打磨台）存在互斥占用约束，还对共享机器人资源存在互斥约束，且工业工序（搬运 transport、打磨 polishing、焊接 welding、倒角 beveling、装配 assembly 五类）具有严格的顺序与资源依赖。

- 输入场景 $S = \{\mathbb{R}, \mathbb{M}, \mathbb{W}\}$（JSON 描述）：机器人集合 $\mathbb{R}$（含外设与可达设备）、机器集合 $\mathbb{M}$（含放置工件与多机可达性）、工件集合 $\mathbb{W}$（含类型与加工状态序列）。
- 扩展析取图 $G = \{\mathbb{V}, \mathbb{C}, \mathbb{D_M}, \mathbb{D_R}\}$：顶点 $\mathbb{V}$ 为工序 $O$（含源点 $V_S$、终点 $V_T$），有向合取边 $\mathbb{C}$ 表示同工件工序先后约束，无向机器析取边 $\mathbb{D_M}$ 与机器人析取边 $\mathbb{D_R}$ 表示资源互斥冲突。

### 2. 两阶段核心机制

1. **规则引导的析取图规划（IV-A）**：向 LLM 提供场景 $S$、自然语言指令 $I$ 与通用规则（而非 few-shot 示例），结合 CoT 推理输出工序集 $\mathbb{O}$、分配集 $\mathbb{A}$ 与单工件内顺序 $\mathbb{Q}$。随后将 $(\mathbb{O}, \mathbb{A}, \mathbb{Q})$ 解析为析取图 $G$，并用确定性启发式 **FIFO** 算法（资源冲突按 $\mathbb{O}$ 中出现顺序裁决）求解无环可行调度图 $F$。LLM 无需直接裁决跨工件/跨机器人冲突。
2. **工序过程树引导的程序生成（IV-B）**：先由 LLM 从程序示例集 $\mathbb{E}$ 中按功能切分、合并同实现节点并细化场景分支，构建统一过程树 $T$ 并经人工校验（一次性构建，跨测试场景复用）；每个节点包含 `Index`、`Type`、`Functional description` 与原子技能代码 `Snippet`。在线生成时，给定 JSON 格式 $T$、$S$、$\mathbb{O}$、$\mathbb{A}$，LLM 为每个工序在树中选定从根到叶的唯一路径，拼接节点片段为通用执行函数 $\mathbb{P_E}$（$B \le L$），再生成每道工序的封装函数 $\mathbb{P_W}$ 与入口调用 $\mathbb{P_C}$。
3. **调度执行（IV-C）**：依据调度图 $F$ 的前置依赖触发各工序对应的 Python 执行函数（当前为开环执行，未引入运行时闭环反馈）。

### 3. IMR-Bench 与核验实验数据

- **基准规模（V-A）**：基于 KunWu 平台 [34]，由产线设计专家改编真实工业环境构建 **23 个场景**（每场景 1–7 台机器人）与 **50 个任务**，分为 Single Robot（1 机器人、$\le 5$ 工序）、Simple Multi-Robot（$\le 3$ 机器人、$\le 10$ 工序）和 Complex Multi-Robot（$\le 7$ 机器人、$\le 24$ 工序）。
- **指标**：`OC`（分配集合 $\mathbb{A}$ 的 IoU）、`SE`（仅当 $OC=1$ 且 $F$ 可行时按 makespan 计算，否则为 0）、`Exe`（代码可执行性）、`GCR`（工件符号状态变化召回率）、`SR`（$SE=1$ 且 $GCR=1$ 时为 1）。
- **Table I 主结果**：使用 GPT-4o 时，IMR-LLM 在 Single / Simple / Complex 三档上的 `SR` 分别为 **0.90 / 0.87 / 0.68**，高于 `LiP-O`（0.90 / 0.73 / 0.24）、`LaMMA-O`（0.80 / 0.46 / 0.20）、`LaMMA-S`（0.70 / 0.33 / 0.16）和 `SMART-LLM`（0.50 / 0.20 / 0.00）；切换开源模型 `Qwen3-32B-thinking` 后三档 `SR` 为 **1.00 / 0.87 / 0.60**。
- **Table II 消融（GPT-4o）**：去掉析取图改由 LLM 直接排顺序（`w/order`）或仅出依赖图（`w/dependency`）后，Complex 档 `SE` 从 **0.75** 分别骤降至 **0.07** 和 **0.39**，`SR` 降至 **0.00** 和 **0.36**；去掉过程树改用示例提示（`w/o T`）后，Simple/Complex 档 `Exe` 从 **0.87 / 0.76** 降至 **0.64 / 0.48**，`SR` 降至 **0.64 / 0.44**。
- **真机实验（V-C）**：3 台带末端执行器及支架相机的机械臂协同完成传送带工件搬运；场景 JSON 由专家手工构建约耗时 15 分钟，经仿真校验安全性与正确性后直接部署到真机执行。

# Retriever: Composing the Perception-Reasoning-Action Loop for Long-Horizon Manipulation

- **作者：** Linfeng Zhao, Haojie Huang, Jiayuan Mao, Weiyu Liu, Lawson L. S. Wong, Mykel J. Kochenderfer
- **机构：** Stanford University, Massachusetts Institute of Technology (MIT), Northeastern University
- **版本 / 载体：** arXiv:2607.17213v2 [cs.RO]（2026-09-11 修订，v1 提交于 2026-07-19）
- **原文：** https://arxiv.org/abs/2607.17213（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2607.17213）
- **项目页 / 代码：** http://retriever.systems（项目主页，亦关联 https://openretriever.org、文档站 https://retriever.build、应用示例站 https://golden.retriever.build） | https://github.com/openretriever/retriever（Apache-2.0 License，**已开源**：PyPI 包 `retriever-core` / 导入名 `retriever` 与 `retriever` CLI，包含 Flow/Clock/Sync 编程模型、IR 编译、in-process / multiprocessing / Dora 后端、文档与教程；配套应用机器人示例位于 `openretriever/golden-retriever`）
- **核对范围：** 通读 arXiv:2607.17213v2 全文（Abstract、Sections I–VII、Appendices A–F、Tables I–IV、Figures 1–10 及 Listings 1–4），读取项目页 http://retriever.systems 公开文本，并静态核对官方仓库 https://github.com/openretriever/retriever 的 `README.md` 与目录结构样本。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[Retriever 实体页](../../wiki/entities/paper-agentic-2607-17213.md)

1. **核心问题与动机（Sec. I–II）**
   - 长程开放世界操作需要组合不同频率、不同延迟的闭环模块（例如 30Hz 相机、低频 VLM 规划、~2Hz VLA 技能、200Hz 底层控制）。
   - 现有范式存在两层割裂：强化学习/模仿学习（MDP/POMDP）假设全局同步离散时间步，忽略模型推理耗时与延迟；机器人中间件（ROS / ROS 2 / Dora）提供传输与回调机制，但将对齐、陈旧度（staleness）、分块（chunking）和触发逻辑隐藏在胶水代码中，导致行为依赖线程调度、难以复现和调试。
   - Retriever 的目标是在不替换底层传输中间件（如 Dora、ROS 2）的前提下，为多速率闭环机器人智能体提供显式时间语义与确定性输入消费契约的“窄腰（narrow-waist）”编程模型与运行时。

2. **连续时间异步形式化与理论保证（Sec. III, Appendix B–C）**
   - **时间与流**：全局连续事件时间 $\mathbb{T} = \mathbb{R}_{\ge 0}$，流 $s: \mathbb{T} \rightharpoonup \mathcal{V}$ 为部分函数，分为连续行为 $s_c: \mathbb{T} \to \mathcal{V}_c$ 与离散事件流 $s_e: \mathbb{T} \rightharpoonup \mathcal{V}_e$。时钟 $\mathcal{C}$ 是触发时刻集 $\{t_k\}$ 上的无量纲事件流。
   - **因果流函数（CSF）与 Flow**：$\mathcal{F}: \mathsf{Stream}(\mathcal{I}) \to \mathsf{Stream}(\mathcal{O})$ 满足前缀因果性 $s_{\mathrm{in}}|_{\le t} = s'_{\mathrm{in}}|_{\le t} \implies \mathcal{F}(s_{\mathrm{in}})|_{\le t} = \mathcal{F}(s'_{\mathrm{in}})|_{\le t}$，并在本地时钟刻度 $t_k$ 执行状态转移 $(h_{t_k^+}, y_{t_k}) = f(h_{t_k^-}, x_{t_k})$，输出在声明的提交时间 $c_k$（通常 $c_k=t_k$，变延迟策略可 $c_k>t_k$）向下游可见。有向反馈环必须包含至少一条严格因果边或状态延迟（只读 $<t$ 时刻事件），避免瞬时代数死循环。
   - **Theorem 1 / 3（可表示性）**：任何具有局部有限决策时刻、固定大小记忆、通过确定性同步策略读取已提交历史并满足严格因果反馈的有界记忆确定性策略，均可由包含 `Map`、`Scan`、时钟和确定性同步边的 Retriever 图实现；通过将随机策略的采样种子或输出实现外显为输入流 $a_k = f(z_k, \epsilon_k)$，结论可扩展至随机策略。
   - **Theorem 2 / 4（函数确定性 / 轨迹确定性）**：固定确定性 Flow 代码、同步策略、初始状态、外部输入历史、时钟/提交轨迹（含滞后决策）、严格因果反馈，以及随机调用的已记录种子/输出，则整个网络的输出流唯一确定、与运行时调度交错无关（指离散事件路径确定性，不要求 GPU 浮点按位一致）。

3. **编程模型与三层运行时架构（Sec. IV, Appendix D）**
   - **核心原语**：`@io` 类型化消息、`Flow[I, O]`（`reset()` + `step(inp)`）、运行时钟（`@Rate(hz=...)` 周期驱动或 `@Trigger(...)` 事件驱动）、`Pipeline` 图构建（`.then(dst, sync=...)`）与核心算子（`Map`、`Scan`、`Latest`、`Window`、`Join`）。
   - **滞后与截止时间语义**：`Latest` 支持有界陈旧度检查；时钟可配置超期丢弃、追赶重放、告警或调试报错，运行中的滞后决策须记入日志以支持严格重放。
   - **三层路径**：Python 声明式定义 → 静态类型化中间表示 IR（`pipe.build_ir()`，可序列化为 JSON 并支持可视化与校验）→ 多后端执行（`pipe.step(dt=...)` 进程内单步调试与 `pipe.run(backend="multiprocessing" | "dora")` 异步部署），并支持 `pipe.record(...)` / `pipe.replay(...)` 结合 MCAP / Rerun 离线重放与时间旅行调试。

4. **长程分层操作案例 Retriever-0（Sec. V, Appendix E）**
   - 由 6 个主 Flow 组成：`F_cam`（30Hz 头部/腕部相机与 60Hz 机器人状态）→ `F_belief`（`inspection_done` 触发，维护三值谓词 `{Yes, No, Unknown}` 并输出 `belief_updated`）→ `F_plan`（`replan` 触发 VLM 规划，产出最多 5 个技能的有界条件 `PlanChunk`）→ `F_exe`（执行监视器：不打断当前执行中技能，仅在上下文匹配且进度预测指示交接点时提交下一 `SkillCmd` 并消解 `IF/THEN/ELSE` 分支）→ `F_skill`（~2Hz VLA 技能流，支持微调 $\pi_{0.5}$ 或从头训练多任务扩散策略，输出 `ActionChunk` 与 `SkillProgress`）→ `F_ctrl`（200Hz 控制器回放 `ActionChunk`）。另有可选 `LanguageGoal` 与 100Hz Meta Quest 3 `Teleoperation` 输入流。
   - 底层共享技能策略在两个任务共约 200–300 条完整专家示教（每条 1–3 分钟，共数小时数据）上训练，覆盖闭集技能词表（`OpenDrawer`、`InspectDrawer`、`CloseDrawer`、`Pick`、`Season`、`PlaceBack`、`OpenBag`、`InspectBag`、`Extract` 等）。

5. **实验与评测数据（Sec. VI, Table IV, Appendix F）**
   - **真机实验（双臂 PiperX 平台）**：
     - **Spice search and seasoning**（4 抽屉搜索、8 步里程碑）：`Retriever-0 (full)` 预计总时间 56s、任务进度 73%；`− Plan chunking` 65s / 63%；`− Progress prediction` 12.5%（首步交接即停滞）；`− Belief memory` 20%（反复重搜首个抽屉）；`− Replanning` 15%；`+ Human-in-the-loop` 50s / 100%；纯 $\pi_{0.5}$ 与纯扩散策略均 0%。
     - **Bag retrieval**（可形变购物袋取物与分类、4 阶段里程碑）：`Retriever-0 (full)` 36s / 100%；`− Plan chunking` 37s / 93%；`− Progress prediction` 25%；`− Belief memory` 45s / 66%；`− Replanning` 47%；`+ Human-in-the-loop` 35s / 100%；纯 $\pi_{0.5}$ 41s / 41%；纯扩散策略 40s / 23%。
   - **消息延迟与代码量（Fig. 8a / 10a, Appendix F-A）**：Dora 后端延迟贴近原生 `dora-rs >= 0.3.12`，明显低于多进程 ROS 2 Python/C++ 基线；单进程 Retriever 在小消息上延迟最低（优于单进程 ROS 2 C++ composable nodes）。延迟基准用户脚本规模：Retriever 单文件 117 行，Dora 为 32 + 53 行，ROS 2 Python 为 79 + 95 行（含配置共 4 个文件 191 行）。
   - **可微物理确定性实验（混合弹跳球，Sec. VI-C, Appendix F）**：标准配置（$T=200, \theta=3.0, e=0.8, dt=0.01, p=0.2, K=100$）下，Retriever 产生 1 条唯一轨迹与 1 个唯一路径梯度（$\sigma_g \approx 3.5 \times 10^{-18}$），而到达时间语义 Pub/Sub 基线产生 100 条不同轨迹、35 个不同梯度（$\sigma_g \approx 0.177$，99% 轨迹失配）。高抖动配置（$T=150, \theta=4.0, e=0.85, dt=0.01, p=0.4, K=50$）下，Retriever 仍保持 1 条唯一轨迹与确定性梯度，Pub/Sub 产生 50 条轨迹（$\sigma_g \approx 0.110$，98% 失配，梯度符号甚至反转）。

# A Physical Agentic Loop for Language-Guided Grasping with Execution-State Monitoring

- **作者：** Wenze Wang, Mehdi Hosseinzadeh, Feras Dayoub
- **机构：** Australian Institute for Machine Learning (AIML), Adelaide University, Australia
- **版本 / 载体：** arXiv:2604.07395v1 [cs.RO]（2026-04-08 提交，6 图，4 表）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2604.07395) / [arXiv Abs](https://arxiv.org/abs/2604.07395) / [项目页](https://wenzewwz123.github.io/Agentic-Loop/)
- **项目页 / 代码：** 宣称将开源或未列可运行代码（官方项目页 [Agentic-Loop](https://wenzewwz123.github.io/Agentic-Loop/) 链接至 GitHub 仓库 [WenzeWWZ123/Agentic-Loop](https://github.com/WenzeWWZ123/Agentic-Loop)，截至 2026-11-27 核查该仓库仅含 `README.md` 与 `docs/` 静态项目页文件，`README.md` 标注 `🚧 Code coming soon`，尚无机器人控制、Watchdog 或实验脚本源码；另附 YouTube 演示视频 [youtu.be/BaT5lDUizZE](https://youtu.be/BaT5lDUizZE)）
- **核对范围：** arXiv:2604.07395v1 HTML 全文（Abstract、§I–§VI、Table I–IV、Fig. 2–6 文字说明与参考文献）及官方项目页文本、GitHub 仓库 `README.md` 与目录样本
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题与思路（§I, §III-A）：** 现有语言引导抓取系统多采用单次开环执行，抓取失败（抓空、滑落、脆弱抓取、堵转、超时或抓错语义目标）不会以结构化状态回传决策层。受数字工具调用智能体（如 OpenClaw [1]、ReAct [3]、Toolformer [4]）启发，论文将语言引导抓取形式化为有界具身智能体 $\mathcal{A} = (\mathcal{O}, \mathcal{S}, \mathcal{U}, \pi)$，在不微调、不重训底层抓取原语（ForceSight [2]）的前提下，通过物理工具状态流（physical tool-state stream）与确定性有界决策策略实现闭环恢复。
- **架构与事件接口（§III-B）：**
  1. **Agent Core：** 使用轻量本地语言模型 TinyLlama-1.1B [30] 解析用户自然语言指令为结构化目标约束 $\mathcal{G}$（类别、颜色、空间限定词），通过感知条件化（颜色滤波或实例分割屏蔽非目标区域）向抓取原语提供目标一致的 RGB-D 输入，并维护有界恢复状态机。
  2. **Event-Based Interface：** 每次抓取尝试绑定唯一 `trial_id`，发布结构化事件：细粒度物理状态 `watchdog_label` $\in \{\texttt{SUCCESS}, \texttt{EMPTY}, \texttt{SLIP}, \texttt{WEAK}, \texttt{STALL}, \texttt{TIMEOUT}\}$、粗粒度状态 `exec_status` $\in \{\texttt{SUCCESS}, \texttt{FAIL}, \texttt{WAIT\_CLARIFY}\}$、时间戳及轻量调试元数据。
  3. **Outcome-Aware Grasping Engine：** 封装未修改的 ForceSight 视觉-力目标预测与伺服原语，以及安全控制与 Watchdog 监测层。
  4. **四阶段闭环：** Observe（接收 $\mathcal{G}$ 与条件化 RGB-D）→ Act（执行抓取原语）→ Evaluate（Watchdog 推断离散状态并可选融合语义校验 $v_{\text{sem}}$）→ Decide（输出 `FINALIZE`、`RETRY/RESELECT` 或 `WAIT_CLARIFY`）。
- **Watchdog 稳定化与有界决策策略（§III-C, §III-D, §IV-E）：**
  - Watchdog 融合夹爪力矩/电流动力学、闭合行为与微抬升（micro-lift）证据，并要求短稳定窗口（settle window）一致才发射标签，以抑制纯接触带来的假阳性。
  - 默认有界策略 $\pi$：`SUCCESS` 且 $v_{\text{sem}}$ 一致（若启用）时 `FINALIZE`，否则 `WAIT_CLARIFY`；`EMPTY` 在重试预算内（默认最多 1 次自动重试）触发 `RETRY/RESELECT`，耗尽后转 `WAIT_CLARIFY`；`SLIP`/`WEAK`/`STALL`/`TIMEOUT` 在保守默认设置下直接 `FINALIZE` 安全终止（或按意图需要升级为 `WAIT_CLARIFY`），保证有限步终止。
- **真机实验与关键数值（§IV）：**
  - **硬件平台：** Hello Robot Stretch 移动操作机器人 + 眼在手上（eye-in-hand）Intel RealSense D405 RGB-D 相机；底层抓取原语为未修改的 ForceSight。全过程累计 >290 次真机试验，最终系统配置下完成 180 次对照试验（各场景 $N=10$）。
  - **主对比（Table II，$N=10$）：** Single Target 从基线开环 **80% (8/10)** 升至 **100% (10/10)**；Ambiguity (Color / Spatial) 从 **40% (4/10)** 升至 **80% (8/10)**；Distractor Robustness 从 **0% (0/10)** 升至 **100% (10/10)**；Multiple Identical Targets 从 **10% (1/10)** 升至 **100% (10/10)**。成功试验中最多仅需 1 次自动重试。
  - **组件消融（Table III，歧义场景 $N=10$，含 Diff-color 5 次与 Same-color 5 次）：** Full System **10/10**（5/5 + 5/5）；No Retry **9/10**（5/5 + 4/5）；No Watchdog **7/10**（3/5 + 4/5）；No Vision Conditioning **4/10**（1/5 + 3/5）。
  - **Watchdog 检测精度（§IV-C, Fig. 4）：** 空抓样本中正确检出 `EMPTY` **43/50**；成功抓取样本中有 **5/50** 被误判为 `EMPTY`。
  - **时间开销（Table IV，单目标 $N=10$）：** 基线均值 **14.78 s**（标准差 2.97 s），Agentic loop 均值 **15.94 s**（标准差 7.27 s）。
- **局限性（§V）：** 当前默认策略偏保守（仅 `EMPTY` 自动重试 1 次，其余异常状态均直接终止，尚未引入针对 `SLIP`/`WEAK` 的差异化恢复）；语义消歧仍受限于上游视觉分割/滤波可靠性；Watchdog 现阶段重在区分空抓与非空抓；超出目标存在性与类别约束的可行性推理仍有限。
- **对 wiki 的映射：** [A Physical Agentic Loop for Language-Guided Grasping with Execution-State Monitoring](../../wiki/entities/paper-agentic-2604-07395.md)

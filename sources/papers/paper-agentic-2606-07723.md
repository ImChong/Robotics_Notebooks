# VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation

- **作者：** Siyi Chen*, Hugo Hadfield, Alex Zook, Mikaela Angelina Uy, Chan Hee Song, Erwin Coumans, Xuning Yang, Faisal Ladhak, Qing Qu, Stan Birchfield, Jonathan Tremblay†, Valts Blukis†（* NVIDIA 实习期间工作；† Project Leads）
- **机构：** NVIDIA, University of Michigan
- **版本 / 载体：** arXiv:2606.07723v1 [cs.RO]（2026-06-05 提交），HTML 全文（CC BY 4.0）；项目页标注 CoRL 2026
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2606.07723) / [arXiv 摘要页](https://arxiv.org/abs/2606.07723) / [项目主页](https://chicychen.github.io/VoLo/)
- **项目页 / 代码：** 已开源 — 官方编排器仓库 [`NVlabs/VoLoAgent`](https://github.com/NVlabs/VoLoAgent)（Apache-2.0，包含 `vlm_orchestrator` 代理、GraspGen/SAM3/Molmo2/cuRobo 工具服务端与多基准评测客户端）及配套基准任务包仓库 [`NVlabs/RoboVoLo`](https://github.com/NVlabs/RoboVoLo)（126 个 RoboLab 任务、123 个 USD 场景与资产包，代码及自有资产为 Apache-2.0，外部 34 类 SimReady 资产需另行同意 NVIDIA 协议下载）
- **核对范围：** 通读论文 HTML 全文 Abstract、§1–§6、Table 1–3、Figure 1–7、References [1]–[79] 及全部附录 Appendix A–K（包含工具架构与四协议传输 B.1–B.2、DLS 多起点 IK 公式与参数、VLM 提示词模板 C.1–C.4、15 类任务与 501 资产细节 D–E、配对符号翻转置换检验与 Wilson 区间 F、Franka FR3 真机 14 任务明细 G、LIBERO/RoboCerebra/VLABench 对比验证 H、分套件消融表 Table 7 I、Sankey 失败流 J、世界/VLM 两轴失败分类与 Table 8 K），并核对项目主页全文与两个官方 GitHub 仓库样本档案
- **通读：** 已完成

## 原文核对笔记

- **问题定义——物理编排（Physical Orchestration，§1, §4.1）：** 虚拟 AI 智能体假设世界在推理期间静止，而物理机器人面对持续演化的现实环境，必须同时完成**监测—中止—重定向（monitor–halt–redirect）**：检测实际场景与当前预期状态之间的偏差、在发生偏离时尽快中止正在执行的动作（并在需要时安全悬停/驻留），随后决定继续、重规划、重写指令或切换工具。
- **VoLoAgent 系统设计（§4.2, Fig. 3, App. B–C）：** 将单一 VLM（默认 Claude Opus 4.6）作为统一编排器，将 VLA/WAM（如 $\pi_{0.5}$、DreamZero）与视觉感知模型（GroundingDINO、SAM2、SAM3、Molmo2）及经典动作原语（`grasp(target)`、`place(destination)`）一同视为**可中断工具**，通过三项机制实现闭环：
  1. **P1 异步工具执行：** 机器人策略/动作原语以 15 Hz（8 步动作块）异步运行，VLM 监控循环以 0.2 Hz 并发执行，不阻塞底层运动；
  2. **P2 快慢双记忆上下文：** 轻量监控上下文仅包含当前观测（前视相机 + `BEFORE`/`NOW` 图像）、当前子目标与近期决策；完整审议上下文保存任务目标、初始场景、完整子目标栈与工具目录，仅在初始规划与异常恢复时调用；
  3. **P3 安全驻留（Safety-aware idling）：** 触发恢复时先令当前在飞工具安全停驻（单臂固定底座下即保持静止），再由 VLM 从 `continue`、`replan`、`rewrite`、`grasp`、`place` 中选择纠正动作。
- **工程代理架构与动作原语细节（App. B）：**
  - 四进程隔离代理：`Eval Client`（RoboLab/LIBERO/RoboCasa/VLABench）↔ `VoLoAgent Orchestrator`（规范化 OpenPI msgpack-numpy Schema，通过 Frontend/Backend 插件适配 OpenPI WebSocket、GR00T ZMQ、OpenVLA REST、File-IPC 四种协议）↔ `VLA/WAM Server`（15 Hz）与 `Tool Server`（HTTP RPC 托管 GraspGen + SAM3/SAM2/GDino + Molmo2 + cuRobo）。
  - `grasp(target)` 链路：开放词汇检测/分割 → 深度图点云裁剪 → GraspGen 6-DoF 抓取候选 → 多起点阻尼最小二乘 IK（$\lambda = 5\times 10^{-3}$，零空间增益 $\alpha=0.5$，单步 $\|\Delta q\|_2 \le 0.2$ rad，收敛阈值位置 $<5\times 10^{-4}$ m、旋转 $<5\times 10^{-3}$ rad，上限 500 次迭代并带腕部 $\pm\pi$ 翻转与随机重采样 + $<2^\circ$ FK 校验）→ 输出与 VLA 同构的 8-D 动作块。
  - `place(destination)` 链路：Molmo2-point 或 VLM-point 输出 2-D 像素点 → 深度反投影为 3-D 目标点 → 优先保持夹爪当前姿态或回退至自上而下姿态求解 IK 释放。
  - 互补性机制（§4.2, §5.2）：当 VLA 抓错物体时，`grasp` 工具将夹爪重新定位到正确目标上方，即便 `grasp` 在接触阶段失败，也能为后续重新接管的 VLA 提供干净视野与就近位姿。
- **RoboVoLo 基准（§3, Table 4, App. D–E）：** 基于 RoboLab 与 NVIDIA Isaac Sim / Isaac Lab（PhysX + PathTracer 渲染）构建，采用 DROID 配置（7-DoF Franka FR3 + Robotiq 2F-85 夹爪 + 外置 ZED 2i / 腕部 ZED mini / 前视 ZED 2i，动作空间为 7-DoF 关节位置 + 1-D 二值夹爪，224×224 图像）。在 RoboLab 资产库基础上新增 501 个带碰撞几何与物理材质的物体（247 个 Lightwheel SimReady 日用品 + 118 个元素周期表方块 + 120 个几何艺术构件 + 16 个数学数字/运算方块），覆盖 4 大套件、15 个类别共 126 个长时程任务（Common Sense 32、Memory 30、Complex References 32、World Knowledge 32），每任务跑 3 个固定种子试验（每套系统全量评测约需 18 GPU-hours）。
- **核心实验结果（§5, Table 1–3, Table 5–8）：**
  - **主实验（Table 1）：** 在 4 个 RoboVoLo 套件和 Robolab-Vague（120 任务）上，VoLoAgent (Full) 分别取得 **54.44%**（Common Sense）、**36.90%**（Memory）、**51.04%**（Complex References）、**25.00%**（World Knowledge）和 **31.94%**（Robolab-Vague）成功率，全面高于单动作模型（$\pi_{0.5}$ 为 11.11% / 13.10% / 16.67% / 9.38% / 16.94%；DreamZero 为 13.33% / 15.85% / 4.17% / 3.12% / 18.61%）、Code-as-Policy（CaP-X 最高 15.56% / 15.48% / 7.29% / 2.08% / 14.44%）以及 TAMP（TiPToP 为 3.33% / 1.19% / 20.83% / 22.92% / 18.89%）。
  - **跨套件消融（Table 2 & Table 7，126 任务均值）：** $\pi_{0.5}$ 纯 VLA 12.57%、VoLoAgent (No VLA) 17.76%、VoLoAgent (Only VLA) 34.97%、VoLoAgent (Full) 41.80%。更换感知后端（GDino+SAM2/Molmo2 38.52%，SAM3/VLM-point 36.07%，Exterior camera 36.94%）仍保持稳健；更换 VLM 显著影响上限（Claude Opus 4.6 41.80% > Claude Sonnet 4.6 36.34% > GPT-5.5 35.52% > GPT-5-mini 34.70% > Gemini-2.5-Flash 31.97% > Qwen3-VL-8B 19.95%）；将编排器用于其他 VLA 均带来 2–6 倍提升（$\pi_0$-FAST 26.23%、MolmoBot 24.86%、DreamZero 21.86%、MolmoAct2 12.57%）。
  - **失败模式审计（§5.3, Fig. 5–6, Table 8）：** 90 个 Common Sense 试验中，VoLoAgent 无故障回合数为 $\pi_{0.5}$ 的 5 倍（20 vs. 4），发生故障后的恢复率达 54%（38/70）vs. 13%（11/86）。VLM 侧错误在所有模型上均由完成度监控（Completion monitor）主导（占比 >67%）：Claude Opus 4.6 共 102 次错误（其中 Completion 84、Failure-mon 2、Tool-use 12、Planning 4），Qwen3-VL-8B 达 456 次（Completion 359、Failure-mon 89）。
  - **真机验证（§5.5, Table 3 & Table 5）：** 在物理 DROID 单元（Franka FR3 + Robotiq 2F-85）的 14 个任务 × 3 试验（共 168 次 rollout）中零样本迁移：$\pi_{0.5}$ 14.3%（6/42，95% CI [6.7, 27.8]）、VoLoAgent (No VLA) 45.2%（19/42）、VoLoAgent (Only VLA) 40.5%（17/42）、VoLoAgent (Full) 42.9%（18/42，95% CI [29.1, 57.8]）。
  - **其他仿真环境发现（App. H, Table 6）：** 在 LIBERO 自建 10 个复合任务上，移除语言输入（no prompt）的 $\pi_{0.5}$ 谓词满足率（PSR）仍有 30.0%，与显式指令（35.0%）相差仅 5 pp，说明窄分布微调的仿真 VLA 对场景过拟合而忽略指令；同时非写实 MuJoCo 渲染导致 GroundingDINO/SAM3/Molmo2 明显失效。
- **局限（§6, App. A）：** 完成度监控误差率仍高；云端 VLM 单次调用延迟 1–5 秒限制高频快速纠偏；当前工具原语限于单臂平行夹爪桌面操作，且安全驻留假设静态停驻即可保安全（不直接适用于双足平衡人形机器人）。

对 wiki 的映射：[VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation](../../wiki/entities/paper-agentic-2606-07723.md)

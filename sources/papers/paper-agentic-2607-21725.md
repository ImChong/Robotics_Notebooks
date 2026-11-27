# Addressing the Orchestration Gap in Generalist Robots via Physical Agency

- **作者：** Liane Galanti, Dhruv Shah, Tri Dao
- **机构：** Princeton University, Together AI
- **版本 / 载体：** arXiv:2607.21725v1 [cs.RO]（2026-07-23 提交，arXiv HTML 全文）
- **原文：** https://arxiv.org/html/2607.21725 （摘要页：https://arxiv.org/abs/2607.21725 ；项目页：https://lianegalanti.github.io/Pigey/ ）
- **项目页 / 代码：** https://github.com/lianegalanti/Pigey （已开源；公开 `real/` 真机 TypeScript 编排器与 `sim/` LIBERO 仿真 Python 编排器及系统提示词，上游依赖 `openpi`、`tiptop`、`DROID`、`LIBERO` / `LIBERO-PRO` 均为公开仓库；仓库根目录未附带独立 LICENSE 文件）
- **核对范围：** 全文通读 arXiv HTML（第 1–6 节正文、Tables 1–3、Figures 1–4 及全部附录 Sections 7–18，含 Tables 4–15、Algorithm 1–2、Tool Schemas 与 Prompt Template），并核对官方 GitHub 仓库主页、README 与目录结构（`real/agent.ts`、`real/agent-system.md`、`sim/agent_sim.py`、`sim/spatial_tools.py`、`sim/README.md`）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[Addressing the Orchestration Gap in Generalist Robots via Physical Agency](../../wiki/entities/paper-agentic-2607-21725.md)

### 核心命题：编排鸿沟（Orchestration Gap）

- 作者将固定策略权重在「直接给定完整自然语言指令 $\pi(o, I)$」与「经由推理时闭环智能体编排调用」之间的成功率差值定义为 **orchestration gap**。
- 在不采集新机器人演示数据、不对底层 VLA 或规划器做任何微调的前提下，仅通过推理时物理智能体 **Pigey**（Physical Agency）补齐高层规划、子目标分解、后端路由、状态记忆、执行验证与失败恢复，即可大幅释放冻结技能的潜在能力。

### 架构与闭环机制（Sections 3, 9–11, 14, 16）

- **编排器映射**：$\phi: \mathcal{O} \times \mathcal{H} \times \mathcal{I} \to \mathcal{T}$，每步输出一个工具调用，从不直接发送连续电机动作；底层策略 $\pi_b: \mathcal{O} \times \mathcal{S} \to \mathcal{A}_{\mathrm{motor}}$（$b \in \{\textsc{tamp}, \textsc{vla}\}$）只接收短具体子目标 $s \in \mathcal{S}$（如 `"put the red cup on the plate"`），不接收抽象顶层指令 $I$。
- **双互补冻结后端（真机 DROID 平台）**：
  1. **TAMP 后端（`Pick` / `DropAbove`）**：基于 TiPToP（Gemini Robotics-ER 检测 + SAM 2 分割 + FoundationStereo 深度 + M2T2 抓取预测 + cuRobo/cuTAMP 规划），将原 TiPToP 的单次开环 pick-and-place 拆分为两步工具，中间插入抓取验证。
  2. **VLA 后端（`VLARollout`）**：冻结 $\pi_{0.5}$-DROID，每 $H=8$ 步预测长度为 15 的 action chunk，默认单次 rollout 执行 $K=300$ 个控制步，处理柔性体（线缆、布料）、容器内部/堆叠物体以及 TAMP 失败后的升级恢复。
- **双重验证与确定性失败注解**：观察通道 $o_t = (I_t^{\mathrm{ext}}, I_t^{\mathrm{wrist}}, x_t)$ 中的 $x_t$ 为控制器与日志计算的确定性谓词（末端位姿、夹爪开度、`is_grasped` 布尔标志、无抓点/规划失败/不可达/步数耗尽等）。若后端返回成功但 `is_grasped` 为假，系统强制覆盖为失败。
- **系统级不变量与盲测隔离**：
  - `Done` 的紧邻前一步必须为 `Perceive`，否则被编排器拒绝，防止未核验提前终止。
  - `Pick`/`DropAbove` 的 `label` 必须与最近一次 `Perceive` 返回的已知物体列表精确匹配，避免检测器被幻觉目标名引导。
  - 长时程记忆任务（LM1/LM2）通过 `LookAway`（基座旋转约 180° 使腕部相机背离桌面）和 `LookBack(verify_only=true/false)`（仅在验证遮挡结束时轮询第三视角）在物理层防止信息泄露。

### 关键实验与消融核对（Sections 4–5, 12–13, 17–18）

- **LIBERO-PRO 仿真（每套 6 组扰动，每任务 10 次试验，共约 5,400 次；使用 7 个仿真工具 `Perceive`, `Grasp`, `Place`, `VLARollout`, `VerifyCandidate`, `GoHome`, `Release`）**：
  - $\pi_0$-LIBERO：均值 0%；$\pi_{0.5}$-LIBERO 直接推理：均值 12.8%（Obj swap 17%, Obj task 1%, Sp swap 20%, Sp task 1%, Goal swap 38%, Goal task 0%）；CaP-Agent0：均值 18.2%。
  - Pigey + $\pi_{0.5}$-LIBERO（Claude Opus 4.7）：均值 **53.3%**（Obj swap 54%, Obj task 54%, Sp swap 66%, Sp task 80%, Goal swap 44%, Goal task 22%）。
  - 9 个前沿 VLM 推理器消融（Table 15）：均值覆盖 44.3%（GPT-5.5 low）至 53.3%（Claude Opus 4.7），包含 GPT-5.5 med (46.3%)、Claude Haiku 4.5 (47.7%)、Gemini Rob-ER 1.6 (48.0%)、Gemini 3.5 Flash (48.0%)、Gemini 3.1 Pro (48.7%)、GPT-5.5 high (49.0%)、Claude Sonnet 4.6 (51.7%)。
- **DROID 真机（Franka FR3 + Robotiq 2F-85，30 个任务 × 5 次试验 = 150 次/方法，使用 Claude Opus 4.7）**：
  - 整体成功率：$\pi_{0.5}$-DROID 为 **16.7%**（125 次失败，其中 86 次 grounding 错误、38 次推理/规划错误、1 次抓取失败）；TiPToP 为 **48.7%**（77 次失败，其中 65 次推理/规划错误、7 次抓取失败、5 次 grounding 错误）；Pigey 为 **97.3%**（仅 4 次失败：2 次抓取执行失败 + 2 次验证器假阳性 `verifier false-success`）。
  - 8 类能力探针对比（$\pi_{0.5}$ / TiPToP / Pigey）：Simple pick-place (95% / 80% / 100%)、World knowledge (0% / 90% / 100%)、Conditional logic (0% / 95% / 100%)、Multi-step reasoning (0% / 25% / 100%)、Spatial reasoning (20% / 75% / 100%)、Obstacle/Safety reasoning (0% / 0% / 90%)、Error recovery (10% / 0% / 90%)、Long-horizon memory (0% / 0% / 100%)。
- **开销与局限（Sections 6, 15, 18）**：真机单次试验调用 3–15 次推理器，API 费用约 $0.02–$0.50/次，墙钟耗时 2–6 分钟；受限于底层技能支持集、遮挡导致的不完全可观测假阳性以及前沿 VLM 远程调用延迟。

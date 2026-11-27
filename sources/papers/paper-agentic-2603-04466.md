# Act-Observe-Rewrite: Multimodal Coding Agents as In-Context Policy Learners for Robot Manipulation

- **作者：** Vaishak Kumar
- **机构：** General Bionix, Inc.
- **版本 / 载体：** arXiv:2603.04466v1 [cs.RO]（2026-03-03，arXiv HTML 全文，CC BY 4.0）
- **原文：** https://arxiv.org/html/2603.04466 （摘要页：https://arxiv.org/abs/2603.04466 ）
- **项目页 / 代码：** 未确认开源代码（arXiv v1 正文与检索材料均未给出官方项目页或 GitHub 仓库链接；原文仅在实验说明中使用 `Claude Code` 与 `robosuite v1.5.1`）
- **核对范围：** 已通读 `arXiv:2603.04466v1` 全文，包括 §1 引言、§2 问题建模、§3 相关工作与 Table 1、§4 AOR 框架与 Figure 1、§5 视觉与控制器实例化（含式 (1)）、§6 实验设置与 Table 2 / Table 3、§7 讨论、§8 局限与未来方向、§9 结论，以及 Appendix A（3 个诊断案例与系统弱点分析）。
- **通读：** 已完成

## 原文核对笔记

- **核心问题与定位：** 提出 Act–Observe–Rewrite（AOR）双时间尺度上下文策略学习（in-context policy learning）框架：不更新神经网络权重、不需要人类演示、不做奖励工程，而是让多模态编码智能体（实验中为 `Claude Code`，`claude-sonnet-4-x` 系列）在试验回合之间阅读关键帧图像与结构化轨迹结果，直接重写完整的低层可执行 Python 控制器与视觉处理代码。
- **双时间尺度闭环（§4，Figure 1）：**
  - **快环（回合内控制频率）：** 单视角 $256 \times 256$ RGB-D 相机图像经视觉流水线转为特征字典 $\mathbf{f}_t = \{\texttt{object\_pos}, \texttt{eef\_pos}, \texttt{detected}, \dots\}$，输入 Python `Controller` 类（`reset()` 与 `get_action(features)`），向 `robosuite`（UR5e + `OSC_POSE`）输出控制动作；不读取仿真器真值物体位姿。
  - **慢环（回合间推理频率）：** 回合结束后将结果 $\tau = \{r, T, \phi_1, \dots, \phi_k\}$（奖励、步数、阶段日志、最小末端到物体距离、振荡标志）及阶段切换/等间隔关键帧 $\mathcal{I}_n$ 存入持久情景记忆；多模态 LLM 先回答三问（主导失败模式、根因属视觉/控制逻辑/参数、单次最有效改动），再生成新版控制器代码。
- **安全与稳定机制（§4.2）：**
  1. **编译沙箱：** 独立命名空间编译与接口校验，编译或实例化失败时回退至上一可用控制器。
  2. **动作限幅与异常保护：** 笛卡尔增量限幅 $[-2, 2]$、夹爪限幅 $[-1, 1]$，单步运行时异常输出零动作（safe-stop）。
  3. **有界重写（Bounded rewrites）：** 要求围绕已诊断失败模式做局部定向修改，避免丢弃先验结构的全盘重写。
- **自发演化的视觉与控制设计（§5）：**
  - **OpenGL 相机约定修正：** `robosuite` 默认 `IMAGE_CONVENTION = "opengl"`（第 0 行为图像底部，$y$ 向上、$z$ 向后），标准 OpenCV 反投影使用 $-y_p$ 及 `get_camera_extrinsic_matrix` 的 OpenCV 轴修正会带来 5–8 cm 误差；智能体在 `Stack` 第 6 轮自主定位出 `src/vision.py` 中反投影 $y$ 翻转问题与外参需直接由 `cam_xmat` 计算的问题，将定位误差降至 `cubeA` 1.4 cm / `cubeB` 1.9 cm。
  - **控制策略规律：** 形成阶段状态机 `reach -> descend -> grasp -> lift -> {task-specific}` 与超时回退；自主发现两项原则：（1）抓取闭合期间末端保持静止（stationary grasp），避免下压力推偏物体；（2）指数移动平均动作平滑 $\mathbf{a}_t = \alpha \hat{\mathbf{a}}_t + (1-\alpha)\mathbf{a}_{t-1}$，$\alpha \in [0.3, 0.5]$。
- **实验结果（§6，Table 2–3）：**
  - **Lift：** 3 次 LLM 调用达到 100% 成功率（4/4 随机重复全部成功）。
  - **PickPlaceCan：** 2 次 LLM 调用达到 100% 成功率（20/20 随机重复全部成功）；第 1 次修正在 `agentview` 中可乐罐呈红色而非银色的 HSV 检测，第 2 次改用最大连通域过滤排除料箱红色标记干扰。
  - **Stack：** 经 20 个控制器版本（v0–v20）达到 91% 成功率：v0–v6 调参 0% -> v7–v8 修正反投影 $y$ 翻转 33% -> v9 修正外参 45% -> v10 校准后视觉 70% -> v11–v16 回退斜坡与下降调参 ~75% -> v17 增加抓取重试退化至 6/10 -> v18 共享 `x_correction` 导致放置偏移 76% -> v19 分离 `x_corr_A`/`x_corr_B` 83% -> v20 进入放置阶段冻结 `cubeB` 估计达 91%。剩余 9% 为下降放置 `cubeA` 时夹爪指尖碰移 `cubeB`，智能体正确诊断出根因但未在预算内找到避碰放置策略。
  - **Coding Agent 对比：** 同预算下将 `Claude Code` 替换为 OpenAI `Codex`（`GPT 5.3`，High reasoning）后，三个任务均未能求解（包括最简单的 `Lift`）。
- **Appendix A 揭示的三类典型弱点：** 重构引入副作用（如 v18 修改共享参数影响下游阶段）、随机抓取波动与系统偏差混淆、置信度严重失准（0.8+ 诊断导致性能回退而 0.5–0.6 带来显著提升）。
- **对 wiki 的映射：** [Act-Observe-Rewrite 实体页](../../wiki/entities/paper-agentic-2603-04466.md)

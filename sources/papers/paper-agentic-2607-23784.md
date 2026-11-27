# A Few Words Go a Long Way: Language Guided Robot Policy Synthesis

- **作者：** Daphne Chen, Archit Ritesh Jain, Eric Goossen, Emma Romig, Michael Murray, Nick Walker, Maya Cakmak
- **机构：** University of Washington、Microsoft Research、Massachusetts Institute of Technology（致谢 Amazon Robotics 资助）
- **版本 / 载体：** arXiv:2607.23784v1 [cs.RO]（提交于 2026-07-26）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2607.23784) / [arXiv HTML 全文](https://arxiv.org/html/2607.23784)
- **项目页 / 代码：** 已开源；[官方项目页](https://robo-architect.github.io/)、[官方代码仓库 `robo-architect/architect-franka`](https://github.com/robo-architect/architect-franka)（MIT，含智能体策略合成端 `architect/` 与 Franka Panda ROS 1 Noetic 工具服务栈 `src/franka_robot_apis/`）；注：外部推理模型服务（cuRobo、Grounded-SAM2、AnyGrasp、AnyPlace、DIFT、VLM）通过 WebSocket/HTTP 对接，不在该仓库内提供权重。关联仓库归档见 [paper-agentic-2607-23784](../repos/paper-agentic-2607-23784.md)。
- **核对范围：** 通读 arXiv HTML 全文（Abstract、Sections 1–7、References、Appendices A–H 含 Tables 1–6 及 Figs. 1–7 说明文字）、官方项目页文本，以及 `robo-architect/architect-franka` README 与目录结构。
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与定位：**
   - 黑盒 VLA 模型（如 $\pi_0$、$\pi_{0.5}$、MolmoAct2、GR00T N1.7）对分布偏移和指令措辞敏感，且缺乏可解释的纠偏接口（微调 7B VLA 往往需数百至上千条演示及数小时 GPU 训练）。
   - 传统一次性代码合成方法（如 Code as Policies、ProgPrompt）受限于**规范缺口（specification gap）**：初始自然语言指令难以覆盖物理与几何细节，开环生成后难以定位局部故障并跨任务累积经验。
   - **ARCHITECT**（Agentic Robot Code with Human In ThE loop CorrecTions）将策略获取建模为**人机协同的交互式程序合成任务**，无需任何机器人演示数据或模型微调。
2. **智能体合成与工具套件（Section 3 & Appendices F, H）：**
   - 编排智能体（实验中采用 Claude Opus 4.6）按固定工作流调用元工具：`list_primitives()` → `query_robot_state()` 与 `get_scene_description()` → 上游感知（`detect_objects()` 仅用于待抓物体，`get_placement_pose(base, target)` 用于放置目标）→ `write_primitive(name, code, docstring)` 注册模块化子操作 → `submit_program(code)` 提交可读顶层程序（默认采用基于 `get_vqa_response()` 的“VQA 校验-重试”模式）。
   - 暴露 18 个底层机器人原语（Tables 4–6）：
     - **控制（7 个）：** `set_gripper_width()`（Robotiq 2F-85 0–0.085 m）、`move_ee_to_pose()`（cuRobo WebSocket 规划 + 可选轴向力保护与 reflex 恢复）、`move_ee_to_rel_pose()`、`move_ee_guarded()`（单轴接触停机）、`execute_waypoints()`、`rotate_wrist()`、`reset_robot()`。
     - **感知（8 个）：** `detect_objects()`（Grounded SAM2 + AnyGrasp）、`get_placement_pose()`（AnyPlace）、`get_vqa_response()`（Table 5 标注 GPT-5.4；仓库支持 Qwen VLM 或 OpenAI/Azure OpenAI）、`get_keypoints()`（DIFT）、`get_keypoints_trajectory()`（DIFT + ReKep 阶段化路径点）、`verify_grasp()`（夹爪宽度门控 + 腕部相机 VLM 双重验证）、`get_object_mask()`（Grounded SAM2 无损 NPZ）、`get_grasp_config()`（独立 AnyGrasp）。
     - **本体感觉（3 个）：** `get_current_ee_pose()`、`get_current_joints()`、`get_gripper_width()`。
3. **执行轨迹诊断与持久化技能库（Sections 3.2–3.3 & Appendix G）：**
   - 当执行失败或回合结束时，人类监督者以自然语言提供纠偏（无需懂代码或机器人术语）；智能体结合**程序执行轨迹（execution trace）**定位失败代码行/工具调用并重新合成程序。
   - 纠偏进一步蒸馏为持久化**技能库（Skill Library）**中的可复用技能条目（含 `When to use` / `How` / 代码模式 / `What to watch for`），作为长期上下文学习（long-term in-context learning）在后续任务的每次代码生成前注入上下文。Appendix G 示例 `Guarded Place on Surface` 将用户的口语要求转化为先 `move_ee_to_pose` 再 `move_ee_guarded(axis="z", distance=-0.25, force_threshold=5.0)` 后开爪的固定模板。
4. **真机基准与人工评测结论（Sections 4–5 & Appendices A–E）：**
   - 硬件平台：Franka Panda（DROID 配置，Robotiq 2F-85 夹爪、腕部 ZED-mini 与场景 ZED 2i 相机）+ cuRobo。
   - **8 项真机操作基准（每任务 $N=10$ 回合；ARCHITECT-HiTL 最多 5 次带人类纠偏的 rollout 后评测最终策略）：**
     - 简单抓取放置（Banana → plate `0.80`、Block → basket `0.70`、Baseball → bucket `1.00`）；
     - 复杂任务优势显著：多步序列（Apple & Bread → bowl）`0.80`（vs CaP `0.10`、$\pi_0$/$\pi_{0.5}$ `0.40`）、杂乱盒子取面包（Clutter）`0.70`（vs 基线 $\le 0.40$）、推合抽屉（Articulated）`0.90`（vs $\pi_0$ `0.10`，其余 `0.00`）、对角折布（Deformable）`0.80`（vs ARCHITECT-VLM `0.70`，其余 `0.00`）、布下取蕉（Long Horizon）`0.80`（vs $\pi_0$ `0.10`，其余 `0.00`）。
   - **指令语言扰动消融（Appendix B Table 3，Apple → pan，15 种指令变体）：** ARCHITECT 在 13/15 种扰动下保持 `SR=1.00`（仅在超长描述和缺少放置目标的 `"Move the apple."` 上失败），而 $\pi_{0.5}$ 仅在 6/15 种下成功。
   - **$N=6$ 被试内人工评测（Task 1 立放杯子 → Task 2 货架立放意面盒）：** 载入 Task 1 技能库后，Task 2 的平均人工纠偏查询次数从 `4.67` 降至 `0.83`（$p=0.036$，3/6 人零纠偏）；无试验期纠偏的零样本迁移成功率从 `0/6` 升至 `4/6`。
   - **主要局限（Section 6 & Appendix E）：** 依赖底层模块能力上限（首要失败源于 AnyGrasp 抓取采样与深度估计误差导致的放置高度偏差，以及 cuRobo 碰撞规划失败）；迁移效果依赖先前任务中积累纠偏的密度与覆盖度。
5. **对 wiki 的映射：** 归纳为实体页 [A Few Words Go a Long Way: Language Guided Robot Policy Synthesis](../../wiki/entities/paper-agentic-2607-23784.md)。

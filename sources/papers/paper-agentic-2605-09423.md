# SimWorld Studio: Automatic Environment Generation with Evolving Coding Agent for Embodied Agent Learning

- **作者：** Haoqiang Kang\*, Xiaokang Ye\*, Yuhan Liu, Siddhant Hitesh Mantri, Lingjun Mao, James Fleming, Drishti Regmi, Lianhui Qin
- **机构：** UC San Diego (1)、New York University (2)
- **版本 / 载体：** arXiv:2605.09423v2 [cs.AI]（2026-05-13；v1 提交于 2026-05-10；License: CC BY 4.0）
- **原文：** <https://arxiv.org/html/2605.09423>（arXiv HTML v2 全文）；<https://arxiv.org/abs/2605.09423>
- **项目页 / 代码：** 宣称将开源但未确认可运行代码 — 论文给出仓库地址 <https://github.com/SimWorld-AI/SimWorld-Studio> 并注明「upon acceptance」，截至 2026-11-27 核查该 GitHub URL 返回 HTTP 404（`exists: false`）。
- **核对范围：** 通读 arXiv:2605.09423v2 HTML 全文（2743 行文本），涵盖 Abstract、§1 Introduction、§2.1 SimCoder、§2.2 Co-Evolution、§3.1–§3.3 三个 Case Study（Table 1、Table 2、Fig. 1–9）、§4 Related Work（Table 3）、§5 Conclusion，以及 Appendices A–I（Appendix C.1 Table 4 的 14 个 MCP 工具参考、C.2 五阶段生成流水线、C.3 技能库结构与 5 个内置技能、C.4 集装箱迷宫 4 轮运行案例、C.5 GUI 说明、C.6–C.7 硬件/软件需求 Table 5 与资产许可 Table 6、E.1–E.3 分难度表 Table 7/指标表 Table 8/消融表 Table 9、F.1–F.5 8×H100 实验细节与模态消融 Table 10、G.1–G.3 结构化 GEPA 规则演化与 L0–L7 难度表 Table 11、H–I 提示与定性示例）。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [SimWorld Studio（arXiv:2605.09423）](../../wiki/entities/paper-agentic-2605-09423.md)

### 1. 研究动机与核心定位

- **与静态 3D 场景生成器的本质区别：** 现有具身仿真平台（AI2-THOR、Habitat、CARLA 等）依赖手工场景，ProcTHOR/Infinigen 受限于模板，而近期 LLM 3D 场景生成器（如 Holodeck、SceneCraft 等）多产静态视觉资产，缺少可执行任务、奖励信号和标准学习接口。
- **平台定位：** 基于 Unreal Engine 5.3.2 与 [SimWorld](https://arxiv.org/abs/2512.01078)（Ye et al., 2025）资产/运行时构建，通过自演化编码智能体 **SimCoder** 将文本/图像/编辑指令自动转为可交互、可验证、导出为 Gymnasium（`reset()` / `step(action)`）的 3D 具身学习环境，并支持环境生成器与具身学习者之间的**协同演化（Co-Evolution）**。

### 2. SimCoder 架构与机制（§2.1 & Appendix C）

- **14 个 MCP 工具（Table 4）：** 分为四组：
  1. **Actor 管理（8 个）：** `spawn_blueprint_actor`、`spawn_actor`、`delete_actor`、`delete_all_spawned`、`get_actors_in_level`、`find_actors_by_name`、`set_actor_transform`、`take_screenshot`。
  2. **环境与资产（2 个）：** `setup_environment`（初始化光照/天空/雾/地面/视距，必须在生成 actor 前调用）、`list_assets`。
  3. **场景评估（3 个）：** `verify_scene`（VLM 接收 6 张多角度截图、原始请求与场景图，返回 `PASS` / `NEEDS_IMPROVEMENT` / `FAIL`）、`check_collisions`（世界空间 AABB 重叠检测，返回相交面积 $\text{cm}^2$）、`check_vertical_support`（悬空检测）。
  4. **Python 逃生舱（1 个）：** `execute_python_script`（在 UE5 内置 Python 解释器中执行任意 `unreal` 脚本；可复用的模式经自演化晋升为一等工具或技能）。
- **技能库（Skill Library，Appendix C.3）：** 位于工具之上，采用带 YAML frontmatter 的 Markdown 文档 + 可选配套 Python 工具函数（`python_util`）。预置 5 个内置技能：`building-placement`（覆盖 127 类建筑的尺寸与间距表）、`city-layout`、`street-furniture`、`weather-and-mood`、`screenshot-tour`。
- **五阶段流水线（Appendix C.2）：**
  1. 上下文获取（`list_assets` + `SkillRegistry` + `setup_environment`）；
  2. 上下文内布局规划（不显式外化文件）；
  3. 增量场景构建与每步规则校验（`check_collisions` + `check_vertical_support`）；
  4. VLM 语义校验（`verify_scene`，最多 3 轮修正，在 `PASS` 或达上限时终止）；
  5. 技能编写（自演化：将反复出现的修正策略泛化并写回技能目录）。
- **任务生成与 Gymnasium 编译：** 基于 UE5 NavMesh 采样可达起终点并通过 A\* 计算测地最短路径 $L^*$，筛选 $3\,\text{m} \le L^* \le 20\,\text{m}$ 的回合，实例化 PointNav（目标坐标）与 ObjectNav（语义物体类别，要求沿最短路径至少在一个航点可见）。导出标准 Gymnasium 接口，返回 $224\times224$ RGB（及 RGB-D）观测、目标方位角、测地距离、已走步数与奖励；离散动作空间为 `move_forward`（0.25 m）、`turn_left`（15°）、`turn_right`（15°）、`stop`（Case Study 2 最多 40 步；Case Study 3 超时阈值 500 步）。

### 3. 协同演化机制（§2.2 & Appendix G）

- **双向闭环：** 具身智能体在当前批次环境中训练/尝试，将其表现经三层反馈通道（场景级、结果级、轨迹级）回传给 SimCoder，SimCoder 在不更新底层 LLM 权重的情况下，通过上下文重加权工具/技能调用或调节难度参数生成下一批贴近学习者能力边界的环境。
- **Case Study 3 具体实现：**
  - **具身智能体侧（Structured GEPA 规则累积）：** 与每轮重写整个系统提示词的 GEPA 不同，本方法维护有序优先规则列表 $\mathcal{R}_t = [r_1, \dots, r_k]$，按五类失败模式（方向错误、循环行为、目标邻近失败、避障失败、超时）合成增量规则 $\mathcal{R}_{t+1} = \mathcal{R}_t \cup \Delta\mathcal{R}(\mathcal{F}_t)$，上限 30 条并剪枝连续 20 个 episode 未激活的规则。
  - **环境难度侧（L0–L7 掌握度门控）：** 沿路径长度（400–800 cm 至 2500–3500 cm）、初始航向偏差（0–15° 至 0–180°）和动态障碍密度（0.00 至 0.35）划分 8 级（Table 11）。当最近 5 个 epoch 平均成功率 $\bar{S}_t \ge \tau_{\ell_t}$ 时晋级，阈值 $(\tau_0,\dots,\tau_7) = (0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45)$。

### 4. 关键定量结果核对

- **Case Study 1（场景生成，3 设定 × 3 难度 = 9 场景，Claude Code 脚手架）：**
  - **Table 1（平均分）：** Claude Opus 4.6 在 S1 Text-to-Scene（0.77）、S2 Image+Text-to-Scene（0.79）、S3 Scene Editing（0.75）均居首位；Sonnet 4.6 为 0.70 / 0.73 / 0.74；Qwen3.5-27B 为 0.59 / 0.67 / 0.52；Qwen3.5-9B 为 0.36 / 0.45 / 0.34。Opus 4.6 与 Sonnet 4.6 的无碰撞率均 $\ge 0.98$。
  - **组件消融（Fig. 5）：** 基础代码智能体 0.16 → +MCP 工具 0.45（+0.29）→ +校验循环 0.55（+0.10）→ +自演化 0.76（+0.21）。Appendix Table 9 给出分难度消融。
- **Case Study 2（生成环境训练导航智能体，1,200 训练回合 / 329 测试回合 + 外部 SimWorld-MMNav）：**
  - 采用 L1（步级）、L2（轨迹级）、L3（任务级，每 10 回合蒸馏）三层非参数记忆（Appendix F.4），在 8×H100 服务器并行 8 个 SimWorld Studio 实例（每遍 1.2K 回合约 8 小时）。
  - **Table 2（329 测试回合）：** Qwen3.5-27B PointNav SR 20.21% → 26.76%，ObjectNav SR 22.56% → 31.59%；Qwen3.5-9B PointNav SR 0.40% → 7.43%，ObjectNav SR 0.00% → 14.62%；Qwen3.5-2B PointNav SR 1.38% → 6.23%，ObjectNav SR 0.24% → 4.58%。
  - **多样性与迁移（Fig. 8）：** 固定 200 回合预算下，环境数从 1 增至 30 使 Qwen3.5-27B 测试成功率提升 +5.5 pp；迁移至外部 SimWorld-MMNav 上 Qwen3.5-2B 提升 +12.0 pp。
  - **模态消融（Table 10）：** 100 训练 / 30 未见测试回合上，Qwen3.5-27B 在 RGB（Unseen SR 0.833）、Depth（0.667）、RGB+Depth（0.667）均高于 Text only（0.500）；9B 上仅 RGB+Depth 在 Unseen SR 取得非零成功率（0.167）。
- **Case Study 3（协同演化，Qwen3.5-9B，25 epochs / 500 回合，评估于 329 回合 SimWorld-MMNav）：**
  - 协同演化（L0–L7 自适应课程 + 规则累积）最终测试 SR 达 **90%**，较固定难度（L3，**72%**）高 **+18 pp**，较未训练基线（**50%**）高 **+40 pp**。

### 5. 系统要求与局限

- **运行门槛（Table 5）：** Linux（推荐 Ubuntu 20.04+）、8GB+ VRAM NVIDIA GPU（实测 L40S/T4/A100）、驱动 525+（支持 Vulkan）、Node.js 18+、Python 3.9+、40 GB 可用磁盘（15 GB 下载 + 21 GB 解压）；依赖 UE5.3.2、SimWorld 资产库与 Claude/Qwen 模型。
- **论文自述局限（Appendix A）与核对局限：** 生成质量仍受限于底层编码智能体的 3D 空间推理与精细多物体布局能力；Case Study 2/3 主要围绕导航任务和 LLM 上下文记忆/规则演化验证，未展开复杂物理接触操作（manipulation）或大规模端到端梯度微调（如 PPO/VLA）的实测；官方 GitHub 仓库当前 404，源码运行时序尚未可独立验证。

# From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?

- **作者：** Shinas Shaji, Fabian Huppertz, Alex Mitrevski, Sebastian Houben
- **机构：** 波恩-莱茵-锡格应用科学大学智能与自主系统研究所（Institute of AI and Autonomous Systems ($A^2S$), Hochschule Bonn-Rhein-Sieg）、查尔姆斯理工大学系统与控制部（Chalmers University of Technology）、弗劳恩霍夫智能分析与信息系统研究所（Fraunhofer IAIS）
- **版本 / 载体：** arXiv:2603.03148v1 [cs.RO]（2026-03-03 提交）；已录用于 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
- **原文：** [arXiv:2603.03148 (HTML v1)](https://arxiv.org/html/2603.03148) / [arXiv 摘要页](https://arxiv.org/abs/2603.03148)
- **项目页 / 代码：** 已开源 — 官方代码仓库 [ShinasShaji/llm-robot-cognition](https://github.com/ShinasShaji/llm-robot-cognition)（README 另附论文网页 [icra-llm-robot-cognition](https://shinasshaji.github.io/icra-llm-robot-cognition/)；仓库提供 PyBullet 仿真世界、工具接口、ChromaDB 情景记忆、模型加载器、批量评测与绘图脚本；仓库未声明显式开源许可证文件）
- **核对范围：** 通读 `arXiv:2603.03148v1` HTML 全文（Abstract、§I Introduction、Fig. 1、§II Related Work、§III Methodology §III-A–§III-C、Fig. 2、Listing 1、§IV Evaluation §IV-A–§IV-E、Table I–II、Fig. 3–5、§V Discussion、References [1]–[36]），并核对官方 GitHub 仓库 [ShinasShaji/llm-robot-cognition](https://github.com/ShinasShaji/llm-robot-cognition) 的文件清单与 README 说明。
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?](../../wiki/entities/paper-agentic-2603-03148.md)

### 1. 研究目标与系统架构（§I, §III）

- **核心问题：** 探讨零样本（zero-shot）使用的 LLM 是否能够作为具身机器人认知架构的核心决策引擎，实现任务规划、记忆利用与执行失败恢复。
- **仿真环境（§III-A）：** 基于 PyBullet 构建两室一廊家居环境（厨房含三层货架与餐桌，客厅含桌上的电视，中间以走廊连通），配语义导航图。机器人为全向移动底盘 + 7-DOF 机械臂（派生自 PyBullet KUKA LBR `iiwa` 模型，取消底座旋转关节限位实现 $360^\circ$ 可达）；抓放通过末端进入邻近阈值后直接附着/释放物体简化，未实现多指灵巧手。
- **世界交互工具集（§III-B，共 8 个工具）：**
  1. `Look-Around`：返回当前语义位置与附近物体列表及摆放位置（直接读取仿真状态，不处理视觉图像）。
  2. `Move-To`：基于 A* 算法在语义地图上规划并导航至目标点。
  3. `Grab`：控制机械臂靠近物体，在阈值内附着；若物体不存在、不可达或已持物则返回失败。
  4. `Place`：到达指定位置且该位置空闲且手中有物时释放物体；否则返回违反条件。
  5. `Add-To-Scratchpad`：将推理与计划写入草稿本。
  6. `View-Scratchpad`：查看已有草稿本内容。
  7. `End-Task`：提交任务描述、模型自判状态（成功/失败）及执行动作摘要，并写入情景记忆。
  8. `Search-Memory`：按当前任务描述语义检索情景记忆中的过往经验。
  - 与传统符号规划器不同，调用 `Grab`/`Place` 后**不**维护显式符号知识库更新，世界状态演化完全由 LLM 隐式跟踪。
- **双层记忆与推理机制（§III-C）：**
  - **工作记忆：** LLM 上下文 + Scratchpad（保存可能超出上下文或用于显式分步规划的文本笔记），采用 Chain-of-Thought（CoT）与函数调用。
  - **情景记忆：** 基于 ChromaDB 向量数据库存储历史 `End-Task` 报告，支持文本嵌入相似度检索。
  - **执行监控与重规划：** 工具内部检查执行前/后置条件并返回文本状态；失败后的重规划并非程序强制，而是由 LLM 推理自然涌现。

### 2. 实验设置与定量结果（§IV-A–§IV-D）

- **测试任务（初始状态固定）：**
  - **T1（物品归位）：** `"Put away all the items into the shelf."` — 将 `mug`、`box`、`cube` 三个物品放入三层 `Kitchen Shelf`。
  - **T2（物品交换）：** `"Swap the mug and the cube in the shelf."` — 仅在 T1 完成后执行（测试长时程与单臂一次仅持一物的临时中转推理）。
- **被测模型：**
  - 云端 API：GPT-4.1、Claude 4 Sonnet；
  - 本地集群量化部署：Qwen3 Coder 480B A35B Instruct（vLLM + 4-bit AWQ）、DeepSeek V3.1（`llama.cpp` + 2-bit Unsloth Dynamic 2.0 GGUF）。
  - 预测试中约 24B 参数量的较小模型无法连贯串联动作或陷入无限生成循环，未纳入正式对比。
- **真实执行成功率（Tab. I & Tab. II，以仿真引擎真实世界状态判定；连续 10 次拒绝执行视为失败）：**
  - **T1：** GPT-4.1 `100%`（84 次）、Claude 4 Sonnet `100%`（18 次）、Qwen3 Coder 480B `80%`（400 次）、DeepSeek V3.1 `100%`（311 次）。
  - **T2：** GPT-4.1 `44.4%`（81 次）、Claude 4 Sonnet `100%`（16 次）、Qwen3 Coder 480B `66.2%`（320 次）、DeepSeek V3.1 `75.5%`（306 次）。
  - **响应延迟：** Claude 4 Sonnet 虽然两项任务均达 100%，但单次请求耗时 `30–150 s`，难以满足实体机器人实时响应要求。
- **成功幻觉与情景记忆影响（Fig. 3–5）：**
  - 除零失败的 Claude 4 Sonnet 外，其余模型均会在实际失败时产生“自认为成功”的幻觉（Fig. 3），其中 DeepSeek V3.1 在所有试验中均自判成功。
  - 对比无记忆与累积至多 3 次记忆（Fig. 4）：对成功率的影响因模型和任务而异（GPT-4.1 与 DeepSeek V3.1 略有上升；Qwen3 Coder 在 T1 微升但在 T2 略降），原因在于情景记忆直接存储 LLM 自判状态，幻觉标签会污染后续检索。
  - 工具调用次数（Fig. 5）：累积情景记忆在各模型和任务上平均均可减少工具调用步数，提升执行效率。

### 3. 定性发现与讨论局限（§IV-E, §V）

- **语义可供性缺失与字面误解：** 例如在 T1 中多个模型尝试把客厅桌上的大型 `TV` 抓去放进货架，触发失败后仍能正确完成 `mug`/`box`/`cube` 的归位。
- **顺序任务拒绝执行：** 完成 T1 后紧接着执行 T2 时，模型可能产生“任务已完成”的幻觉并连续拒绝执行（10 次调用后判失败）。
- **未来改进方向：** 引入外部事实核查模块（人类或独立判别模型）校验 `End-Task` 真实状态后再写入记忆；扩展视觉感知、真实机械臂抓取规划与真机部署验证。

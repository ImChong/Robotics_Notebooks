# MeMoREval 论文来源归档

## 元数据

- **题目：** What Should Embodied Agents Remember? Benchmarking and Evaluating Memory Modality for Embodied Agents
- **方法/基准：** MeMoREval
- **作者：** Junxiang Wang, Yuchen Liu, Yi Wu, Guofei Chen, Nir Pechuk, Tao Chen, Zackory Erickson, Yuzhe Qin, Tianyu Li
- **单位（以项目页/HTML 为准）：** Dexmate；Carnegie Mellon University；Purdue University（部分作者为 Dexmate 实习期间完成）
- **预印本：** https://arxiv.org/abs/2610.12561 （v1，2026-10-08，cs.RO）
- **HTML：** https://arxiv.org/html/2610.12561v1
- **项目页：** https://MeMoREval.github.io/
- **代码：** https://github.com/jimwang418/MeMoREval （截至 2026-10-12 为空仓库，未发布；arXiv 摘要页未列代码链接，项目页与 HTML 给出该链接）
- **核对日期：** 2026-10-12

## 论文要解决的问题

LLM 具身智能体需要把过去的视觉/空间观测存起来，供之后未知的任务使用。可存的形式很多（文字记录、生成的图像描述、原始图像帧、场景图），它们决定后续任务能用到什么信息，也决定写入和读取各要花多少推理成本。此前缺少把「记忆模态」单独隔离出来、同时衡量成功率与推理成本的协议。

## 协议与基准设计

- **流程：** 智能体先经历一段事件序列（不知道后续用途），之后在没有新观测、新上下文只剩记忆的前提下回答信息类请求或提交动作类请求（开环技能计划）。
- **仿真：** Genesis，场景改编自 SceneSmith；四个室内环境（办公室、摄影棚、住宅、诊所），每个 7–9 个房间、200–300 个物体。机器人为 Dexmate Vega 1 双臂移动操作平台，通过导航、开关家具、抓放等技能执行。
- **任务：** 53 个单任务、七类：物体位置 11、空间配置 11、外观绑定 10、集合聚合 8、用户偏好 5、遮挡状态 4、技能结果 4；另有每环境 10–11 个任务串联的堆叠组合任务。
- **指标：** 任务/信息/动作成功率；写入与读取的加权 token 成本（并折算美元）。读取每请求最多 5 次检索调用，每轮最多渲染 20,000 字符文本记忆。
- **记忆模态：** Text（写入模型把文本观测整理成记录，丢弃颜色纹理）；desc@64/256/1024（用同一模型对带标注帧生成不同 token 预算的描述，逐步存储）；Frames（带检测框和整数 ID 的标注帧原样归档，按步号检索）；Frames+graph（帧加场景图快照：物体 ID、世界系 3D 位置、支撑/包含边）。另设无记忆与全信息（手写事实）两个对照。
- **记忆后端（5 种）：** transcript、wiki、A-MEM、Mem0、ObsMem（后三者适配到统一接口）。

## 论文报告的结果

主模型 Gemini 3.7 Flash，六个条件、五个后端、53 任务、三个随机种子（温度 0）池化结果：

| 条件 | 任务成功 | 信息成功 | 动作成功 | 写入 k tokens/步 |
|---|---|---|---|---|
| Text | 39% | 53% | 51% | 2.48 |
| desc@64 / @256 / @1024 | 56% / 59% / 60% | 70% / 73% / 73% | 64% / 68% / 71% | 2.95 / 3.14 / 3.22 |
| Frames | 70% | 82% | 77% | 2.48 |
| Frames+graph | 75% | 85% | 82% | 2.48 |
| 无记忆 / 全信息 | 6% / 100% | 17% / 100% | 0% / 100% | – |

- **跨模型（种子 0、三个后端）：** Frames+graph 对三个模型都是总体最高——Gemini 3.7 Flash 72%、GPT 5.6 Sol 57%、GLM 5.3 Flash 50%（Text 分别为 40/36/33）。
- **成本：** 对主模型，Frames 与 Frames+graph 写入成本约 $9.30/千步，与 Text 相同；各描述条件为 $11.08–$12.09。读取每次成功信息请求 Frames $0.026，desc@1024 $0.030。
- **类别差：** Frames 相对 desc@1024，外观绑定 +18.0 pp、物体位置 +12.7 pp、空间配置 +11.5 pp；Frames+graph 相对 Frames，空间配置 +9.7 pp、用户偏好 +8.0 pp。
- **扩展实验：** 堆叠累积经验时 wiki+frames 64%、frames+graph 74%，transcript 从单任务 68% 跌到 28%（历史超出记忆上下文预算）；选择性保留帧（保留 8–55%）在各后端都低于自动全量归档（如 wiki 46% 对 63%）；让写入模型看到图像对成功率无一致收益（如 Mem0 69→60，ObsMem 68→68）。
- **真机：** Dexmate Vega 1，SLAM 建图、人工标注物体与支撑关系生成场景图，头部相机加 SAM 3 分割，wiki 后端 + frames+graph；两个演示（丢弃饮料罐后取回原来旁边的水果；移动柠檬后回忆盘子颜色并取柠檬右侧的苹果）均无人干预完成，仅为定性展示，未报告量化指标。

## 复现与解释边界

- 完整扫描只覆盖 Gemini 3.7 Flash；GPT 5.6 Sol 与 GLM 5.3 Flash 仅单种子、三个后端。
- 多数结果来自仿真；真机只有两个定性演示，且真机场景图依赖人工标注。
- A-MEM、Mem0、ObsMem 是适配到统一接口的版本，可能不代表原设计。
- 成本用加权 token，只在同一模型内可比；部分类别差的置信区间含 0；论文无独立局限性章节，上述局限由正文推出。
- 代码仓库截至 2026-10-12 为空，基准当前不能由第三方复现。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-memoreval-embodied-memory-modality.md
- 项目页归档：sources/sites/memoreval.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.12561>；HTML：<https://arxiv.org/html/2610.12561v1>
- 项目页：<https://MeMoREval.github.io/>；代码：<https://github.com/jimwang418/MeMoREval>

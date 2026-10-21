# HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents

> 来源归档（ingest）

- **标题：** HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
- **类型：** paper / robot-coding-agent / manipulation / benchmark
- **arXiv：** <https://arxiv.org/abs/2610.23479>
- **提交日期：** 2026-10-16（v1）
- **项目主页：** <https://hexafuture.ai>（由官方仓库 README 链出；本次无法读取站点正文）
- **代码：** <https://github.com/HexaFuture/HexaShell>
- **评测基准：** <https://github.com/RoboDojo-Benchmark/RoboDojo>
- **作者：** Hongcheng Gao、Alex Zhou、Jay Zhu、Zelin Zheng、Shijia Ge、Xiping Li、Di Wu、Yazhe Wang、Jianshu Zeng、Xuan Shangguan、Yongli Wang、Yexing Wan、Lingyu He、Zhiqi Jia、Sihang Wu、Xiao He
- **入库日期：** 2026-10-21
- **一句话说明：** 研究视觉语言模型编码 Agent 如何通过单一机器人命令行控制双臂操作，并把失败经验写成可复用 Python 工具；公开论文复测在 RoboDojo 42 项任务上达到 39.7% 成功率、48.0 进度分。

## 核心摘录（面向 wiki 编译）

### 1）薄 Harness：把模型生成代码接到机器人动作

- **方法：** Agent 接收任务指令、三路 RGB 相机图像及标定信息，通过一个命令行调用工具；工具以机器人可观察信息为输入，计算目标位置并执行操作。
- **硬件 / 环境：** RoboDojo 双臂 ARX X5，仿真运行在 Isaac Sim。
- **边界：** 论文称 harness 只负责把 Agent 命令转成机器人动作；不是一个训练出的通用策略网络。

### 2）失败后写工具，沉淀技能

- **方法：** 操作失败后，Agent 读取执行记录、分析失败原因，并创建或修改小型 Python 工具；新工具可扩展命令行，后续任务继续复用。
- **观测约束：** 工具只能使用机器人观察到的信息；论文以双目射线三角测量、已知平面交点和移动腕部相机补视角等方法估计几何关系。
- **开发与测试：** 在每个任务一个 seed 的 10 个布局上开发，再冻结工具并在同一 seed 的另外 10 个布局上测试。

### 3）两组 RoboDojo 数字不可混作同一轮评测

| 结果 | 口径 | 成功率 | 进度分 |
|------|------|------:|------:|
| 论文复测 | 作者机器；42 项任务的新布局测试 | **39.7%** | **48.0** |
| 官方评测 | HexaShell 仓库 README 所述；隐藏 seed、每任务 50 episodes | **46.9%**（截图显示 46.85%） | **53.2** |

官方 README 将后一结果称为 RoboDojo 官方评测榜首。两组结果的评测布局和运行流程不同，不能据此直接判断模型性能变化；引用时须注明口径。

## 对 wiki 的映射

- 主实体：[HexaShell](../../wiki/entities/paper-hexashell.md)
- 基准实体：[RoboDojo](../../wiki/entities/robodojo.md)
- 仓库归档：[sources/repos/hexashell.md](../repos/hexashell.md)

## 参考链接

- arXiv 摘要与论文：<https://arxiv.org/abs/2610.23479>
- 官方代码与结果说明：<https://github.com/HexaFuture/HexaShell>
- RoboDojo 基准代码：<https://github.com/RoboDojo-Benchmark/RoboDojo>

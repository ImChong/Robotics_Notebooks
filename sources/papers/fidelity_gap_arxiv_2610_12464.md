# What 30,000 Hours of Ego-centric Video Does Not Teach

> 来源归档（paper）

- **标题：** What 30,000 Hours of Ego-centric Video Does Not Teach
- **类型：** paper
- **作者：** Jiahua Dong, Anurag Bagchi, Yash Jangir, Muhammad Zubair Irshad, Sergey Zakharov, Martial Hebert, Homanga Bharadhwaj, Yu-Xiong Wang, Vitor Campagnolo Guizilini, Pavel Tokmakov
- **机构：** University of Illinois Urbana-Champaign、Carnegie Mellon University、Johns Hopkins University、Toyota Research Institute（作者单位以论文为准）
- **日期：** 2026-10-08
- **arXiv：** <https://arxiv.org/abs/2610.12464>
- **HTML：** <https://arxiv.org/html/2610.12464v1>
- **项目页：** <https://fidelity-gap.github.io/>
- **代码 / 基准：** 项目页标注 Soon；截至核查日未提供公开仓库或评测集下载链接
- **论文许可：** CC BY 4.0（仅论文许可，不代表训练数据、代码或评测集许可）
- **一句话说明：** 对 Cosmos 3 视频世界模型进行 300–30,000 小时 egocentric 视频规模实验，分别测量手部运动与其对物体交互的结构忠实度；发现扩数据/扩模型更容易拟合动作主体，物体交互仍明显落后，并以 object-centric adaptive noise scheduling 小幅抬高物体保真度上限。

## 核心论文摘录（MVP）

### 1) 动机与度量：主体动作对了，不等于物体响应对了

- **链接：** <https://arxiv.org/abs/2610.12464>
- **摘录要点：** 世界模型若要用于模拟或动作规划，必须预测“动作如何改变被操作物体”，而不只是让画面逼真。作者将手部（agent）与被操作物体的结构一致性分开测量，并用 LPIPS 控制视觉感知质量。SCS 以未来帧中的分割掩码 IoU 评估结构忠实度。
- **评测集：** 独立采集的 OOD 测试集，经筛选含 150 段、每段 32 帧（10 fps）视频，共 11,328 个实例掩码；论文称将公开，但尚无下载地址。
- **对 wiki 的映射：**
  - [Fidelity Gap 实体页](../../wiki/entities/paper-fidelity-gap-egocentric-video.md)
  - [仿真评测可复现性与真实代表性](../../wiki/concepts/sim-vs-real-eval-gap.md)

### 2) 规模研究：数据让手更像，未让交互追上

- **链接：** <https://fidelity-gap.github.io/>
- **摘录要点：** Cosmos 3 在 300、3,000、30,000 小时的人类 egocentric 操作数据上训练；条件包含初始画面与动作，生成之后 16 帧（1.6 秒）。投影人体骨架作为视觉条件后，Nano 仅用约 300 小时就达到无骨架基线约 15,000 小时的 agent fidelity（约 50× 数据效率），但物体 SCS 在 30,000 小时仍低于手部。将模型放大 4×，物体拟合上限只提高约 0.034 SCS。
- **对 wiki 的映射：**
  - [Fidelity Gap 实体页](../../wiki/entities/paper-fidelity-gap-egocentric-video.md)
  - [EgoWM](../../wiki/entities/paper-egowm-egocentric-world-model.md)

### 3) 方法与迁移：把去噪监督移向交互区域

- **链接：** <https://arxiv.org/html/2610.12464v1>
- **摘录要点：** 作者用 D4RT 点轨迹与手部空间先验构造动态区域图，在被操作物体/交互区域提高扩散噪声等级，引导模型更多依赖上下文与动力学而非局部外观复制；组合区域噪声调度与损失重加权后，30k 小时物体 SCS 从骨架条件基线 0.527 提到 0.546（+3.7%），LPIPS 基本不变。11 个双臂仿真任务的人形迁移中，成功判定一致率由 Cosmos 3 的 0.67 提高到 0.81。
- **对 wiki 的映射：**
  - [Fidelity Gap 实体页](../../wiki/entities/paper-fidelity-gap-egocentric-video.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)

## 开源与数据边界

- 项目页将 **Paper Code** 和 **Benchmark** 标注为 **Soon**；当前页面没有代码仓库、模型权重或数据集下载入口。
- 论文称会公开 150 段 OOD 评测集，但截止归档时未提供访问地址；1.15M 个训练片段 / 30k 小时人类视频来自 Mecka 支持，不应推断为本文开放数据。
- arXiv 页面标注的 CC BY 4.0 是论文文本许可；不能据此推断训练视频、代码、模型或评测数据采用同一许可。

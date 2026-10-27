# GENESIS-Handover（arXiv:2610.08003）

> 来源归档（ingest）

- **标题：** Reactive Task-Oriented Robot-Human Handovers via Generative Hypothesis Selection
- **类型：** paper / human-robot interaction / task-oriented handover / mobile manipulation
- **arXiv：** <https://arxiv.org/abs/2610.08003>
- **HTML：** <https://arxiv.org/html/2610.08003v1>
- **项目页 / 官方代码：** arXiv 条目未列出项目页或代码仓库；截至 2026-10-27，未核实到官方公开实现。
- **机构：** 苏黎世联邦理工学院（ETH Zurich）Robotic Systems Lab
- **作者：** Carmen Scheidemann、Andreea Tulbure、Pascal Burkhardt、Marco Hutter（前两位同等贡献）
- **发表状态：** arXiv v1，2026-10-06 提交；页面注明接收于 CoRL 2026
- **一句话说明：** 通过视觉语言模型生成多种任务条件化的手-物交互假设，再依据接收者当前手型选择假设，并在递交过程中实时跟踪手部更新机器人末端目标。

## 核心论文摘录（MVP）

### 1）问题与主要贡献（摘要、§1）

- **摘录：** 既有任务导向递物方法大多预测单一抓取构型，难以表达同一物体和任务下的人类抓握多样性。GENESIS-Handover 将问题表述为多模态手-物交互假设选择：用视觉语言模型先产生多种可行使用方式，再与接收者手部姿态匹配。
- **对 wiki 的映射：** [GENESIS-Handover 实体页](../../wiki/entities/paper-genesis-handover.md)；[操作任务](../../wiki/tasks/manipulation.md)。

### 2）方法与系统接口（§3）

- **摘录：** 输入包括场景 RGB-D、物体名称和交接后的任务描述。系统先重建物体网格，再让语言模型提出多样抓握描述、图像生成模型产生手持物体的候选图像；MANO 手部网格与单目深度估计用于尺度/三维落地，ICP 将合成物体与真实物体对齐，AFT-Handover 提供对齐初始化先验。机器人在交接开始时比较实时手部关节与各候选手型，选出最近候选；随后把选中手-物关系锚定到实时手部位姿，持续更新末端目标。
- **对 wiki 的映射：** [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)；[操作任务](../../wiki/tasks/manipulation.md)。

### 3）真机评测与边界（§4、§5、附录）

- **摘录：** ANYmal 腿式移动操作机器人搭载顶部机械臂、Intel RealSense L515 RGB-D 相机和 Robotiq 2F 夹爪；实时手追踪、人体检测、行为树与运动规划在 Jetson Orin 上，候选生成在外部工作站运行。12 名参与者、5 组物体-任务、两方法各 60 次交接。与 AFT-Handover 启发式基线相比，GENESIS 的“无需调整即可使用”交接从 6/60 提高到 16/60；10/12 参与者认为其任务理解更好。附录对额外运行分析了 85 次 GENESIS 交接，7 次失败；错误涉及抓取不匹配、手部姿态估计、运动规划和 ICP 对齐。
- **对 wiki 的映射：** 重点区分用户主观任务理解、客观无需调整次数与总体失败；样本规模和对象任务覆盖有限，不能外推为广泛部署成功率。

## 当前提炼状态

- [x] 论文元数据和方法摘录
- [x] wiki 实体映射确认
- [x] 项目页 / 代码开放状态核查（arXiv 未链接官方项目页或代码仓）
- [x] 关键指标按论文 HTML 核对
# Awesome Touch 上游 README 通读（技术地图视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/sun-awesome-touch-technology-map.md`）

- **标题：** Awesome Touch（sun254667/awesome-touch，README.md）
- **作者 / 维护方：** Yuanliang Sun（sun254667）
- **类型：** repo / awesome-list / curated-index
- **原文：** <https://github.com/sun254667/awesome-touch>（2026-11 通过 raw.githubusercontent.com 获取 main 分支 README 全文，约 33 KB）
- **一句话说明：** 2025.01–2026.07 触觉操作研究精选清单，按 Surveys、Tactile VLA、视触觉世界模型、Tactile WAM、策略学习、仿真 / Sim2Real、传感器硬件、数据集基准、关键挑战九部分组织。
- **核对范围：** 从头到尾读完 README 全部内容（收录标准、目录、9 个分组的全部条目与 Highlights、数据集 / 基准表、Key Research Challenges、联系方式、许可）。清单性质的文档，没有方法 / 实验可读；清单所指向的各篇论文正文**未**在此归档中阅读，其内容以各论文页为准。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **收录标准：** 优先 ICRA / IROS / CoRL / RSS / NeurIPS / ICML / ICLR / CVPR / ICCV / ECCV 等顶会顶刊；覆盖 Tactile VLA、视触觉世界模型、Tactile WAM、触觉模仿学习、触觉仿真与 Sim2Real、数据集基准；时间窗 2025 年 1 月–2026 年 7 月。
- **结构：** 1 Surveys（3 条）；2 Tactile VLA（2.1 统一 VTLA 框架、2.2 基础模型增强）；3 Visuo-Tactile World Models；4 Tactile WAM；5 策略学习（5.1 模仿、5.2 强化、5.3 扩散）；6 仿真与 Sim2Real；7 传感器与硬件；8 数据集与基准；9 关键挑战。每条给出出处（会议 / arXiv）、Highlights、链接，各组内按时间新到旧排序。
- **VLA 注入触觉的几条路线（2.1 Highlights）：** 自适应决定注入时机与位置（AT-VLA）；残差触觉表征缓解模态塌陷（ResTacVLA）；触觉–力对齐（TaF-VLA）；双系统异步，触觉编码器 100 Hz 流式推理、VLA 按需查询压缩历史（TacMamba）；蒸馏成触觉 token 以便推理期不用触觉传感器（HapticVLA）；不微调基座、用触觉语言模型 + 扩散控制器做精修（VLA-Touch）。
- **世界模型 / WAM（3、4 节）：** OmniVTA 含 OmniViTac 数据集（21,000+ 任务、86 物体）与 60 Hz 反射控制器；ContactWorld 在 12 个接触任务上系统比较，结论为空间结构化 + 时间连续的表征规划最强；Tactile-WAM 用触觉非对称注意力 + VideoClean 掩码防止「触觉污染」；VT-WAM 在统一 flow matching 内联合预测未来视觉、触觉形变与动作；TacPAC 把触觉预测缓存为接触计划并实时纠正动作（5 个真机任务均值 22% → 64%）。
- **策略学习（5 节）：** IL 侧有推理期引导（TouchGuide、ViTaL）、跨模态表征（ViTacFormer）、人类数据（FTF、Grasp Like Humans）、人形全身接触（TACT）；RL 侧仅 1 条（四足触觉 loco-manipulation）；扩散侧有 Reactive Diffusion Policy 慢–快双层、VT-Refine 的「少量示范 + 数字孪生 RL 精调」、EquiTac 的 SO(2) 等变性。
- **仿真与硬件（6、7 节）：** TacSL（Isaac 内 GPU 视触觉仿真，称比此前最优快 200 倍以上）、Taccel（IPC+ABD，18 倍实时加速）、SimTac、Hydrosoft、ETac、Tac2Real、UniVTAC；硬件含 PolyTouch（相机 + 声学 + 周边视觉，寿命至少 20 倍）、OSMO / ART-Glove 触觉手套、FreeTacMan 免机器人采集（1 万条轨迹、50 任务）、TwinTac 数字孪生传感器。
- **数据集 / 基准表（8 节）：** OmniViTac、ContactWorld、ManiFeel、WorldArena 2.0、UniVTAC Benchmark、ObjTac（135K 三模态样本）、Tactile MNIST、ManiSkill-ViTac 2025 Challenge，共 8 项。
- **关键挑战（9 节，README 自述为「对 2025–2026 文献的综合」）：** 触觉污染、模态塌陷、数据稀缺、触觉 Sim2Real、跨模态表征兼容性、时空结构、接触中心保真度、实时推理（约 60 Hz）、预测与执行的时序错位、传感器泛化。
- **与站内技术地图的差异（核对发现）：** README 中的 TACO、TouchThinker、ViTacPhys、TacPAC、HiTac-WAM、FTP-1、WT-UMI、ART-Glove 等条目不在站内 65 条地图内；README 的 TacCoRL、τ 在 2.1 内重复列出两次；README 第 3、4 节条目数（7、7）与站内分组数（6、5）不同，说明站内为较早快照。

## 对 wiki 的映射

- [`wiki/overview/sun-awesome-touch-technology-map.md`](../../wiki/overview/sun-awesome-touch-technology-map.md)
- [`wiki/entities/awesome-touch.md`](../../wiki/entities/awesome-touch.md)

## 参考来源

- 上游 README：<https://github.com/sun254667/awesome-touch>

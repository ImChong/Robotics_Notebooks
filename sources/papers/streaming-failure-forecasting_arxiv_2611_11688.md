# Streaming Failure Forecasting（arXiv:2611.11688）

> 来源归档（paper）

- **标题：** Streaming Failure Forecasting: VLA Failure Monitors That Keep Learning After Deployment
- **作者 / 机构：** Amit Jadhav, Shaurya Beriwala, David Q. Liu, Yanfei Liu（Purdue University Fort Wayne）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.11688>（v1，2026-11-08，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.11688v1>
- **项目页：** <https://sff-research.org/>
- **代码：** 论文称协议、结果开启记录与代码「将在 sff-research.org 发布」（Reproducibility 段）；2026-11-11 核查项目页首页，页内只有站内链接，**未见 GitHub 仓库链接**，故不记 `代码：` 仓库地址。
- **入库日期：** 2026-11-11
- **一句话说明：** 冻结离线失败监测网络，部署后每个回合结束用成功/失败标签以闭式解重拟合最后一层线性读出，并只在成功回合后移动报警阈值。
- **核对范围：** 读 arXiv v1 PDF 全文：摘要、§1–§7（引言、相关工作、问题设定、方法、实验设计、结果、局限与结论）、参考文献列表、附录 A（预注册记录）至 J（Table 3–13、Algorithm 1、Fig. 5–13 的文字与表格内容）。图表经 pdftotext 抽取文字阅读，未逐张查看位图像素；未下载或运行任何代码；项目页只读了首页（静态 HTML），未逐篇读其子页博文。
- **通读：** 已完成

## 开源状态

- 论文未给仓库地址，只说代码与协议「将发布」于项目页。项目页首页未见 GitHub 链接（2026-11-11 静态读取）。结论：**未核实到已开源代码**。
- 评测数据来自他人发布的 ArmnetBench v0.1（Hugging Face `armnet/armnetbench_v01_lerobot_so101`，revision 1db69c8）与 LIBERO / LIBERO-Plus，见论文参考文献 [1] [15] [16]。

## 核心摘录

1. **问题（§1、§3）：** VLA 失败监测器离线训练后冻结；部署后光照、物体、相机、策略版本变化会破坏其排序。真实记录上 4 个 VLA 的冻结监测器 within-task AUROC 从拟合任务上的 0.62 降到切换后的 0.536（§1，Fig. 1）。重校准阈值不能修复排序（§1）。
2. **方法（§4，Fig. 2，Algorithm 1）：** 离线小网络（两个 128 宽隐藏层）固定；`z = [1, ϕ(x), t/T]`（130 维）；每个回合后用带遗忘因子 ρ 的折扣最小二乘、以 L2 锚定离线头 θ̄ 重拟合读出 θ（式 (1)），一次 130×130 求解，不存回合。阈值 τ 只在成功回合后按 `τ ← τ + η(1[M>τ] − α)` 更新（α=0.1，η=0.05）。
3. **仿真（§5、§6.1，Table 1）：** SmolVLA 在 LIBERO 系列上加光照、物体颜色、相机位移、磨损、复合漂移。SFF 在 5 个测试套件中的 4 个高于冻结监测器；主检验 LIBERO-10 差 +0.030 [−0.0004, +0.063]，单侧 Holm p=0.026，双侧区间触及 0；自训 SmolVLA 套件 −0.008（p=0.813）。
4. **真机记录回放（§5、§6.2，Table 2）：** ArmnetBench 的 434 条 SO-101 记录、4 个 VLA（SmolVLA、π0.5、π0、MolmoAct 2）、共 40 次切换；每条运行用对应检查点重算特征。切换后 SFF 池化 0.656 对冻结 0.536（+0.121 [+0.077, +0.162]）；对「在真实切换点完全重置」+0.044（探索性）。
5. **代价与校准（§6.4、附录 C）：** Jetson Orin Nano 单 CPU 核中位 2.18 ms/次策略调用，状态 1.0 MB；仿真 FAR 0.082–0.089，复合漂移下 0.19–0.37；真机因只有 5 个校准回合（α=0.5）不报 FAR。
6. **局限（§7）：** 仅 SmolVLA 一个策略族的仿真、增益 0.02–0.05；真机为回放而非在线运行；漂移前学习有小代价（真机首任务 SFF 0.56 对冻结 0.62，附录 H）；需要每回合都有标签。

## 对 wiki 的映射

- [paper-streaming-failure-forecasting](../../wiki/entities/paper-streaming-failure-forecasting.md)

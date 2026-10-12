# Robot-WATCHDOG 论文来源归档

## 元数据

- **题目：** Robot-WATCHDOG: failure detection through object-centric graph representation
- **作者：** Quentin Rolland, Fabrice Mayran de Chamisso, Jean-Baptiste Mouret
- **单位：** Université Paris-Saclay, CEA, List（Palaiseau）；Inria, CNRS, Université de Lorraine, LORIA（Nancy）；Bleu Robotics（Paris，Mouret）
- **预印本：** https://arxiv.org/abs/2610.13403 （cs.RO，v1，2026-10-09；arXiv 备注 Accepted at CoRL 2026）
- **HTML：** https://arxiv.org/html/2610.13403v1
- **项目页：** https://cea-list.github.io/Robot-WATCHDOG/
- **代码：** 论文与项目页均无代码仓库链接；项目页称新数据集 “released with this work”，但无下载链接（见 [sources/sites/robot_watchdog_project_page.md](../sites/robot_watchdog_project_page.md)）
- **入库日期：** 2026-10-12

## 论文要解决的问题

部署学习到的策略需要实时失败检测，但开放世界中失败难以显式定义，故只用正常（nominal）示范学习。现有方法难以既容忍无害变化、又抓住语义与交互层面的错误（拿错物体、顺序错、放错位置）。

## 方法与训练信息

1. **图构造：** 每个回合由 RGB 视频转为按时间步的物体中心图；节点为任务相关物体与机器人末端，节点特征为绝对 2D 位置与瞬时速度；边为两两关系（相对 x/y 位移、距离、朝向 cos/sin，共 5 维）。检测在首帧用 Qwen3-VL-8B-Instruct（提示任务预期物体），跟踪用 SAM2。
2. **TPC（Trajectory Projection and Correlation，非参数）：** 对每个物体计算到同标签最近专家轨迹的正交投影距离，按专家均值与标准差归一化，并以指数递增权重按时间聚合（近期偏差权重大）；再加相关性惩罚：观测与正常成对 Pearson 相关矩阵的 ℓ1 差异。
3. **GnnT（学习型预测器）：** GATv2 编码空间交互（动态注意力）；过去嵌入的滑窗展平成 token 送入 Transformer（位置编码含相对时间、物体 ID、回合进度）；MLP 预测未来位置（时域 L）；损失为位置 MSE + 边一致性损失，仅用正常示范训练。
4. **共形校准：** 用留出的正常回合按用户容忍度 α 构造随时间变化的阈值；作者使用未做有限样本修正的经验分位数，所以误差水平只是近似。

## 论文报告的结果

- **数据集：** 公共基准 BotFails（保留 4 个刚体任务：Table-setting、Dish storing、Vegetable sorting、Groceries sorting；剔除可变形/流体任务）+ 新数据集（结构化摆桌、传送带废物分拣）；合计 6 个任务，10 个随机种子平均（校准结果的每个种子重抽专家划分、校准划分和网络初始化）。新数据集：Table-setting 100 条专家回合（67,341 帧）+ 21 条测试回合（13,471 帧，15 FPS，ALOHA 机器人，4 路相机）；Waste sorting 100 条专家回合（59,562 帧）+ 51 条测试回合（30,378 帧，30 FPS，SO-100，单顶视相机），失败步占比 0.22。新数据集共 272 个回合。
- **基线：** 轨迹匹配 AP-RMS、AP-Fréchet、Soft-DTW；学习表征 logpZO、lopO（连续归一化流）、AE-Recon；视觉语义 FIDeL（主要的基于图像的检测器）、ResNet18 视觉特征。
- **AUPR / 最大 MCC：** TPC：BotFails 0.811 / 0.751，Watchdog（新数据集）0.810 / 0.719；GnnT：0.623 / 0.488，0.648 / 0.631；最强基线：AP-RMS 0.442 / 0.320（BotFails），AP-Fréchet 0.346 / 0.344（Watchdog）。
- **共形校准后 MCC（α∈[0.80, 0.95] 平均）：** TPC 0.675（BotFails）/ 0.641（Watchdog）；GnnT 0.347 / 0.571；最强基线 FIDeL 0.220（BotFails）、logpZO 0.329（Watchdog）。
- **消融：** 视觉特征在 Table-setting 上失效（MCC 0）；图相对仅轨迹模型在该任务 MCC +23.8 点，Waste-sorting 上结果混合；去掉边对 GnnT 有害；TPC 的投影项单独在 Waste-sorting 最好，组合分数在 Table-setting 最好（MCC 88.8）；预测式比密度式（归一化流）跨种子更稳定。
- **延迟（RTX 3090）：** Qwen 初始化 7.03 s/任务一次；SAM2 22.72 ms/帧；GnnT 19.87 ms/帧；TPC 7.00 ms/帧；SAM2+GnnT 端到端 42.59 ms/帧（23.5 FPS）；SAM2+TPC 29.72 ms/帧（33.6 FPS）。

## 复现与解释边界

- 依赖离散物体抽象，不适合可变形物体、流体或涂装/焊接类任务；依赖 VLM 跟踪流程，遮挡、漏检和身份切换会破坏图；只建模物体质心，无法表达朝向与细粒度位姿。
- 论文内部不一致：正文称 10 个种子，Table 4 报告为 5 个种子；FIDeL 在 Table-setting（Watchdog）只有 9 个种子。本归档引用正文的 10 种子口径并在此注明。
- 失败回合数量未在读到的部分给出（仅给失败步占比 0.22，Waste sorting）。
- 阅读边界：HTML 全文已分两段读完，附录 A、B 中的数字来自读取结果。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-robot-watchdog-object-graph-failure-detection.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.13403>
- HTML：<https://arxiv.org/html/2610.13403v1>
- 项目页：<https://cea-list.github.io/Robot-WATCHDOG/>

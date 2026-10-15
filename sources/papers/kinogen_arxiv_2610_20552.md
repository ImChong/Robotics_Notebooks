# KinoGen: Customizable Humanoid Loco-Manipulation References via Kinematically Grounded Video Generation（arXiv:2610.20552）

> 来源归档（ingest）

- **标题：** KinoGen: Customizable Humanoid Loco-Manipulation References via Kinematically Grounded Video Generation
- **arXiv：** <https://arxiv.org/abs/2610.20552>（Submitted 2026-10-14，v1；PDF：<https://arxiv.org/pdf/2610.20552>）
- **项目页：** <https://kinogen-humanoid.github.io/>（页内 Code 按钮标注 **Code coming soon**；**无代码链接**）
- **作者：** Qi Wu, Lulu Zhao, Sanghyun Hahn, Rishi Yennu, Emmy Yang, Dingqi Zhang, Tao Pang, Jiuguang Wang, Kuan Fang
- **机构（PDF 首页）：** Cornell University；Multiply Labs
- **入库日期：** 2026-10-15
- **类型：** paper / humanoid / loco-manipulation / video-generation / motion-reference / unitree-g1
- **一句话说明：** 用校准渲染的机器人与物体作为视频生成条件，再把同一套机器人与物体模型拟合到两视角生成视频上，直接得到机器人构型空间内的全身、手与物体参考轨迹，无需人体代理与重定向。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** 页面 Code 入口文字为 "Code coming soon"，无可点击的仓库链接；页面中的 github 链接仅为作者个人主页。
- **论文核查：** PDF 中仅有项目页 URL，无代码发布声明。
- **结论：** **待发布**（项目页标注 "Code coming soon"，截至 2026-10-15 无代码链接）。

## 摘录 1：问题与主张（Abstract / §I）

- **痛点：** 人类示教需要重建并重定向到机器人，有身体嵌入误差；遥操作需要大量人力覆盖行为、物体、场景与目标变化；纯文本或图像条件的视频生成与度量任务要求是开环的。
- **主张：** 以机器人与物体的标定渲染作为视频生成条件，再通过拟合同一套模型恢复参考，使参考在构型空间内成立，无需人体代理或重定向。
- **可控性：** 编辑一个机器人-物体关键帧即为一个新渲染状态，可据此重新生成关键帧附近的视频，并将重建约束到目标物体位姿上。

## 摘录 2：实验要点（§IV，数字取自 Table I–III）

| 方法 | 穿透 (cm) ↓ | 脚滑 (cm/s) ↓ | 接触保持 (%) ↑ | 平滑度 (m/s²) ↓ | 下游 RL 成功率 (%) ↑ | 参考偏差 (m) ↓ | 学习曲线 AUC ↑ |
|------|------|------|------|------|------|------|------|
| GenMimic | 3.17 | 9.95 | 13.57 | 0.96 | 31 | 0.16 | 29.92 |
| GenHOI | 10.55 | 6.52 | 43.91 | 1.34 | 11 | 0.15 | 19.90 |
| GRAIL | 3.06 | 5.11 | 69.40 | **0.59** | 42 | 0.13 | 38.73 |
| **KinoGen（本文）** | **2.30** | 4.09 | **91.16** | 0.73 | **57** | **0.09** | **60.74** |

- 消融（Table II）：去掉身体先验后，根加速度 0.37 → 1.01 m/s²，穿透 10.5 → 26.8 mm；单视角时最差视角 IoU 下降且 EPE 上升。
- 关键帧修正（Table III）：不做修正时，放置误差在三个区域为 1.07–14.55 cm；直接末端插值能精确到达目标，但会穿过桌面；本文方法精确满足目标并显著降低穿透（具体数值见原表，未在摘录中逐条核实）。
- 任务：8 个移动操作任务（坐凳、跪地取物、拖地、搬箱上架、抬篮、抬椅、哑铃、篮球等）。
- 硬件：Unitree G1 配 OYMotion ROHand 灵巧手；仿真 Isaac Sim；真机展示四项（坐凳、抬椅、抬篮球、跪地取物）。跟踪策略先用特权观测训练，再蒸馏到本体感知 + 初始物体位姿的策略；物体初始位姿由首帧 RGB-D 与 FoundationPose 估计。

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-kinogen.md`](../../wiki/entities/paper-kinogen.md)：强调「视频生成 → 构型空间拟合 → 关键帧可控」的流水线，并把它与重定向类参考生成区分开。
- 与 [mimic-video](../../wiki/methods/mimic-video.md)、[视频即仿真](../../wiki/concepts/video-as-simulation.md)、[运动重定向](../../wiki/concepts/motion-retargeting.md) 互链。

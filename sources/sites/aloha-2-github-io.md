# ALOHA 2 项目页（aloha-2.github.io）

- **标题：** ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation
- **类型：** site / project-page
- **URL：** <https://aloha-2.github.io/>
- **配套论文 / 技术报告：** [arXiv:2405.02292](https://arxiv.org/abs/2405.02292) — 归档见 [`sources/papers/aloha2_arxiv_2405_02292.md`](../papers/aloha2_arxiv_2405_02292.md)
- **PDF：** <https://aloha-2.github.io/assets/aloha2.pdf>
- **仿真资产：** [google-deepmind/mujoco_menagerie `aloha/`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/aloha) — 归档见 [`sources/repos/mujoco-menagerie-aloha.md`](../repos/mujoco-menagerie-aloha.md)
- **硬件教程 / 装配：** 项目页 **Tutorial** 导航；商用套件与装配文档见 [Trossen ALOHA 2.0 Docs](https://docs.trossenrobotics.com/aloha_docs/2.0/)
- **联系：** aloha-2@googlegroups.com
- **入库日期：** 2026-09-27

## 一句话摘要

Google DeepMind **ALOHA 2 Team** 在初代 ALOHA 上升级夹爪、被动重力补偿、铝型材机架与 **Intel RealSense D405** 腕/顶/仰视相机，并 **开源全部硬件设计 + 装配教程**；同步发布 **系统辨识后的 MuJoCo Menagerie 工位模型**，支撑大规模双臂遥操作采数与仿真策略学习。

## 公开信息要点（截至入库日）

- **动机三角：** 性能与任务范围、遥操作人体工学、机队可维护性/鲁棒性，以支撑「更多机器人 × 更长采数时间 × 更宽任务分布」。
- **相对 ALOHA 1 的硬件改动：**
  - **夹爪：** Leader 低摩擦导轨替代剪刀机构；Follower 低摩擦夹爪减延迟、增输出力；升级指尖 griptape。
  - **重力补偿：** 被动机构（现货件）替代橡皮筋方案。
  - **机架：** 20×20 铝型材；去掉操作员对侧竖框，便于人机协作采数与大道具。
  - **相机：** 更小 D405 + 3D 打印支架；相对消费级 USB 摄像头：**更大 FOV、深度、全局快门、可定制**。
  - **仿真：** Menagerie 中 ALOHA 2 工位 MJCF，含 **11 条真机轨迹系统辨识** 的执行器/摩擦参数。
- **页内板块：** Abstract · Grippers · Frame · Simulation（MuJoCo + Google Scanned Objects 遥操作视频）· Tutorial · Sim。
- **采购：** Trossen Robotics — Aloha Stationary / WidowX / ViperX 套件。

## 开源核查（步骤 2.5，2026-09-27）

| 资产 | 状态 |
|------|------|
| 硬件 CAD / 设计文件 + 装配教程 | **已开源**（项目页 Tutorial；论文写明 open source all hardware designs） |
| MuJoCo ALOHA 2 模型 | **已开源** — `mujoco_menagerie/aloha/`（BSD-3-Clause） |
| 单点 GitHub「aloha2_hardware」仓 | **无独立仓列于页首**；设计与教程托管于项目站 + Trossen 文档链 |
| 训练代码 / 数据集 | **非本页主交付**；采数栈仍常接 ACT / 自研 IL 管线 |

## 为何值得保留

- **硬件 + 仿真成对发布：** 腕部 D405 与 Menagerie 相机内参对齐，是 Sim2Real 视觉模仿的显式锚点。
- **与初代 ALOHA / ACT 的关系：** 同一遥操作范式上的 **fleet-scale** 迭代，而非新算法论文。
- **交叉标定语境：** 四路相机 + 双臂 FK 链，天然接 [手眼标定](../../wiki/concepts/hand-eye-calibration.md) 与 [RealSense D405 规格](realsense-d405-product.md)。

## 对 wiki 的映射

- [ALOHA 2 实体](../../wiki/entities/aloha-2.md)
- [ALOHA（初代）](../../wiki/entities/aloha.md)
- [MuJoCo Menagerie](../../wiki/entities/mujoco-menagerie.md)
- [Intel RealSense](../../wiki/entities/intel-realsense.md)

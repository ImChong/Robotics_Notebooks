# NVIDIA SOMA Retargeter：将人类运动适配到机器人（官方技术博客，2026-11-13）

> NVIDIA Technical Blog 一手资料归档；本文提炼机制与工程事实，不转载正文。

- **英文标题：** Adapt Human Motion for Robots with NVIDIA SOMA Retargeter
- **中文标题：** 使用 NVIDIA SOMA Retargeter 将人类运动适配到人形机器人
- **类型：** blog
- **发布日期：** 2026-11-13
- **作者：** John Malaska（NVIDIA）
- **原文：** https://developer.nvidia.com/blog/adapt-human-motion-for-robots-with-nvidia-soma-retargeter/
- **代码：** https://github.com/NVIDIA/soma-retargeter
- **版本：** SOMA Retargeter v0.2（官方博客所述发布版本）
- **许可核查：** NVIDIA 编写的工具源码 Apache-2.0；仓库自带 BONES-SEED 动作样本另受 NVIDIA Sample Data Evaluation License 约束，不能按 Apache-2.0 重新分发。
- **映射到 wiki：** [SOMA Retargeter](../../wiki/entities/soma-retargeter.md)
- **核对范围：** 通读 NVIDIA 技术博客正文「From human motion to robot joint trajectories」「Sources of human motion」「Adapting motion to different humanoid robots」「Configuring your robot and refining its motion」「Using retargeted motion to train robot policies」「Generating motion in real time with MotionBricks」「Get started」；对照官方 README、interactive-retargeting、batch-retargeting、robot-configurator 文档以及 app/bvh_to_csv_converter.py、soma_retargeter/pipelines/soma_retargeting_pipeline.py 的实现入口。
- **通读：** 已完成

## 机制与输入输出

1. 人体动作可来自 BONES-SEED 动捕、GEM-X 视频 3D 重建或 Kimodo/ARDY 生成；需转换到 **SOMA base skeleton 的 BVH**，不是任意 BVH 直接可用。
2. 按机器人身材和零位姿做几何映射与缩放；使用 Newton（基于 NVIDIA Warp）的 IK，按目标权重求每帧机器人关节姿态。
3. 可通过 Robot Configurator 对齐参考零位姿、身体部位映射和比例；实验性 IK weight optimizer 根据代表性动作调整不同部位目标权重。
4. 按配置可启用接触检测、足部着地修正、肢体稳定等后处理；最后约束关节限位。
5. 输出逐帧机器人 **root 位姿 + 关节角度的 CSV**。官方文档规定平移单位 cm，root XYZ 欧拉角与关节角为度；CSV 本身没有帧率字段，随附加载器默认按 120 FPS 播放。

## v0.2 示例机型

- Unitree G1（示例默认目标）
- Unitree H2
- Booster T1
- AgiBot X2 Ultra
- AgiBot A3T3

v0.2 提供 BONES-SEED 的示例 BVH、参考 CSV 与对应配置。官方 README 所列为 10 条示例 BVH 与 5 机型对应输出；软件支持通过 Robot Configurator 注册额外 URDF/MJCF 机器人。

## 训练边界

- 交互 Viewer 支持加载、重定向、人工检查、保存；无界面批处理可将整个 BVH 目录转换为目标机器人 CSV。
- SONIC 提供 SOMA CSV→G1 训练 motion library 的转换入口；ProtoMotions 可用作物理跟踪训练框架。**本工具生成的是运动学参考，而不是可直接上机的稳定控制策略**，仍需在物理仿真检查平衡、接触、执行器约束和跟踪误差。
- 官方 Robot Configurator 文档说明尚不支持高自由度灵巧手结构；当前重点为人形机体运动。批处理要求同批 BVH 具有匹配的骨架拓扑，错误配置或不兼容源骨架会失败。
- 许可证区分：仓库内第三方机器人资产又有各自的 BSD-3-Clause / Mulan PSL v2 条款，需分别核实。

## 已核对的运行入口

- 交互预览：`uv run python app/bvh_to_csv_converter.py --config assets/default_bvh_to_csv_converter_config.json --viewer gl`
- 无 GUI 批处理：`uv run python app/bvh_to_csv_converter.py --config local-configs/batch-retargeting.json`（先按官方文档准备配置）
- 自定义机器人：`uv run python app/tools/robot_config_generator.py --viewer gl`
- 程序入口 `Viewer.retarget_motion` / `Viewer.batched_retargeting` → `SomaRetargetingPipeline.add_input_motions` → `execute` → `csv_utils.save_csv`；`execute` 中运行 BasePass、可选 PostProcessPass、JointLimitClamper 后返回 CSVAnimationBuffer。

## 官方进一步阅读

- [仓库 README 与许可](https://github.com/NVIDIA/soma-retargeter)
- [交互重定向说明](https://github.com/NVIDIA/soma-retargeter/blob/v0.2.0/documentation/interactive-retargeting.md)
- [批处理说明](https://github.com/NVIDIA/soma-retargeter/blob/v0.2.0/documentation/batch-retargeting.md)
- [Robot Configurator](https://github.com/NVIDIA/soma-retargeter/blob/v0.2.0/documentation/robot-configurator.md)
- [源码：BVH 转 CSV 入口](https://github.com/NVIDIA/soma-retargeter/blob/v0.2.0/app/bvh_to_csv_converter.py)
- [源码：核心重定向管线](https://github.com/NVIDIA/soma-retargeter/blob/v0.2.0/soma_retargeter/pipelines/soma_retargeting_pipeline.py)

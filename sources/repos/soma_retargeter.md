# SOMA Retargeter（NVIDIA 官方开源项目）

> 来源归档；对应 [2026-11-13 NVIDIA 技术博客](../blogs/nvidia_soma_retargeter_2026_11_13.md)。

- **英文标题：** SOMA Retargeter
- **中文标题：** SOMA 人体运动到人形机器人的重定向工具
- **类型：** repo
- **链接：** https://github.com/NVIDIA/soma-retargeter
- **官方文档：** https://github.com/NVIDIA/soma-retargeter/tree/v0.2.0/documentation
- **核对版本：** v0.2（2026-11-19 查阅）
- **许可：** NVIDIA 原创代码 Apache-2.0；BONES-SEED 示例动作另为 NVIDIA Sample Data Evaluation License；第三方机体资源按各自许可证。
- **开源状态：** 已开源。交互 Viewer、批处理、机器人配置器、IK 权重优化工具及 5 机型样例配置公开；不是策略模型权重仓库。
- **入库日期：** 2026-06-08
- **一句话说明：** SOMA BVH → 多机型机器人运动 CSV 的 Newton/Warp IK 工具，含比例缩放、可选接触修正与关节限位后处理，提供交互与无界面批处理。
- **沉淀到 wiki：** 是 → [SOMA Retargeter 实体](../../wiki/entities/soma-retargeter.md)
- **核对范围：** 通读官方 README 的 What the retargeter does、Inputs and outputs、Bundled robot targets、License 等段落；对照 v0.2 官方 interactive-retargeting、batch-retargeting、robot-configurator 文档；检查 app/bvh_to_csv_converter.py 的 Viewer.retarget_motion、Viewer.batched_retargeting、main 入口及 soma_retargeter/pipelines/soma_retargeting_pipeline.py 的 add_input_motions / execute 主执行链。
- **通读：** 已完成

## 核心事实（官方 README + 源码）

- 输入限定为 SOMA base skeleton 的 BVH；一般的不同骨架 BVH 要先规范化。
- 官方 v0.2 样例机型：`unitree_g1`、`unitree_h2`、`booster_t1`、`agibot_x2ultra`、`agibot-a3t3`；包含 10 条 BVH 示例及五种对应 CSV。
- `HumanToRobotScaler` 生成已缩放的 IK 效应器目标；`SomaRetargetingPipeline.execute` 中 BasePass 解 IK，按配置可用 PostProcessPass 处理肢体和接触，最后 JointLimitClamper 限制关节坐标。
- 输出 CSV 含根部平移（cm）、XYZ 欧拉角（度）、机器人驱动关节角（度），无内嵌帧率。随附 CSV 读取器默认 120 FPS。
- Robot Configurator 接受 URDF / MJCF；官方文档明确高 DoF 灵巧手尚未支持。
- 重定向只产出运动学参考，不担保在动力学、执行器或真机上直接可行；需经策略训练和仿真验证。

## 官方运行入口

- GUI：`uv run python app/bvh_to_csv_converter.py --config assets/default_bvh_to_csv_converter_config.json --viewer gl`
- 批处理：`uv run python app/bvh_to_csv_converter.py --config local-configs/batch-retargeting.json`（用户自行准备配置；不能把示例配置中的 `unitree_h1` 与现成机型混为一谈）
- 配置器：`uv run python app/tools/robot_config_generator.py --viewer gl`

## 官方关联

- [SOMA-X](https://github.com/NVlabs/SOMA-X)、[GEM-X](https://github.com/NVlabs/GEM-X)、[Kimodo](https://github.com/nv-tlabs/kimodo)
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions)、[SONIC](https://nvlabs.github.io/SONIC/)
- [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)
- [官方技术博客归档](../blogs/nvidia_soma_retargeter_2026_11_13.md)

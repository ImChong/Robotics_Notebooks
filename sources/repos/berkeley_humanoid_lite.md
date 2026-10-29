# Berkeley-Humanoid-Lite

> 来源归档

- **标题：** Berkeley Humanoid Lite
- **类型：** repo
- **组织：** HybridRobotics
- **链接：** https://github.com/HybridRobotics/Berkeley-Humanoid-Lite
- **项目页：** https://lite.berkeley-humanoid.org/
- **文档：** https://berkeley-humanoid-lite.gitbook.io/docs
- **论文：** https://arxiv.org/abs/2504.17249
- **许可：** MIT
- **星标（截至 2026-07-25）：** ~1442
- **入库日期：** 2026-07-25
- **一句话说明：** 低成本 3D 打印开源人形：整机 CAD、摆线关节、电机/控制参数、BOM、底层控制、Isaac Lab 训练与实机部署。
- **开源状态：** **已开源**（主仓 + 门户 + 文档；执行器为成品电机 + 3D 打印摆线）
- **项目页归档：** [sources/sites/berkeley_humanoid_lite.md](../sites/berkeley_humanoid_lite.md)
- **沉淀到 wiki：** [berkeley-humanoid-lite](../../wiki/entities/berkeley-humanoid-lite.md)

---

## 关节相关要点（策展）

| 项 | 内容 |
|----|------|
| 减速比 | 约 **15:1** 摆线 |
| 电机 | 外转子；约 **14 对极**；力矩常数约 **0.1176 N·m/A** |
| 控制 | 公开电流环 / 位置环 / 速度环等参数 |
| 链路 | 设计 → RL（Isaac Lab）→ 实机 |

## 局限（官方叙事 + 策展）

- 当前 **3D 打印摆线** 在高性能运动中偏脆弱；后续版本可能改成品关节。
- 适合学习与验证，**不适合原样用于重型人形**。

## 对 wiki 的映射

- [Berkeley Humanoid Lite](../../wiki/entities/berkeley-humanoid-lite.md)
- [开源 QDD 执行器项目对比](../../wiki/comparisons/open-source-qdd-actuator-projects.md)

## 官方软件组件（2026-10-27 核查）

原论文平台的 [主仓](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite) 包含 Isaac Lab 环境与任务、策略训练、sim2sim、实机部署和遥操作。README 标注主仓代码 MIT，其他资产 CC BY-SA 4.0；各独立仓库许可应分别核查。

- [机器人描述资产](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets)：URDF、MJCF、USD 与 Onshape 导出脚本。
- [实机底层控制](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Lowlevel)：真机运行代码。
- [电机控制器固件](https://github.com/T-K-233/Recoil-Motor-Controller-BESC)；[SteamVR 遥操作桥](https://github.com/ucb-bar/SteamVR-Bridge)。上述入口由[官方发布页](https://berkeley-humanoid-lite.gitbook.io/docs/releases)列出。
- [Berkeley Humanoids 组织](https://github.com/Berkeley-Humanoids)另列 [Lite V1.5 双臂平台](https://github.com/Berkeley-Humanoids/berkeley-humanoid-lite-arm)、[Lite V2 描述资产](https://github.com/Berkeley-Humanoids/lite_description)、[ROS 2 控制栈](https://github.com/Berkeley-Humanoids/humanoid_control_ros2) 和 [低层通信 SDK](https://github.com/Berkeley-Humanoids/lite-sdk2)。这些是后续平台资源，版本与原论文平台不同，兼容性需按各自说明确认。

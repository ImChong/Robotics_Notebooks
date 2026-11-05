# AuRA 官方仓库

> 来源归档（repo）

- **标题：** AuRA: From Atoms to Bits through Autonomous Robot Interaction
- **链接：** <https://github.com/Yoonkyo/AuRA>
- **论文：** <https://arxiv.org/abs/2610.34452v1>
- **入库日期：** 2026-10-28；源码核对：2026-11-03
- **沉淀到 wiki：** [AuRA](../../wiki/entities/paper-aura.md)

## 开源状态与运行入口

官方 README 将当前 v1 定位为自动采集：相机标定、静态采集与抓取筛选、Franka 抓取抬升录制和 PhysTwin 格式转换；后处理及训练重建标为 v2 待发布。仓库自身代码采用 Apache-2.0；AnyGrasp SDK、许可证、检查点不随仓库分发，Grounded-SAM-2 与权重需另行安装；第三方 fr3 许可状态见官方 THIRD_PARTY_NOTICES。

- 准备入口：`scripts/anygrasp_poses_one_shot.sh` 调用静态采集、点云、AnyGrasp、可执行性筛选脚本。
- 采集入口：`scripts/run_anygrasp_to_lift_one_shot.sh` 调用准备脚本与 `aura/robot/execute_anygrasp_franka_record_lift.py`，并转换原始记录及标定到 PhysTwin 格式。
- 官方硬件：Franka Research 3、三台 OAK-D；使用前按官方文档完成标定。可行性筛选不等于全身碰撞检查，默认抓取成功也没有得到充分检查。
- 核对来源：[README](https://github.com/Yoonkyo/AuRA/blob/main/README.md)、[准备脚本](https://github.com/Yoonkyo/AuRA/blob/main/scripts/anygrasp_poses_one_shot.sh)、[采集脚本](https://github.com/Yoonkyo/AuRA/blob/main/scripts/run_anygrasp_to_lift_one_shot.sh)。

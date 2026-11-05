---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/openwbt.md
summary: "银河通用 开源项目 OpenWBT（具身Agent/规划）：头显和手柄提供视角、手部目标与行走命令，上肢通过逆运动学生成手臂目标，下肢策略负责移动和姿态调整；同一部署入口覆盖MuJoCo、Isaac Sim和真实Unitree G1，使全身遥操作链能够分层检查。…"
institutions:
  - galbot
---

# OpenWBT

## 一句话定义

**OpenWBT** 是 [银河通用](https://github.com/GalaxyGeneralRobotics) 公开的 **具身Agent/规划** 开源项目：头显和手柄提供视角、手部目标与行走命令，上肢通过逆运动学生成手臂目标，下肢策略负责移动和姿态调整；同一部署入口覆盖MuJoCo、Isaac Sim和真实Unitree G1，使全身遥操作链能够分层检查。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第二层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 银河通用 |
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/GalaxyGeneralRobotics、https://github.com/PKU-EPIC |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `OpenWBT` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：真机 `python -m deploy.run_teleoperation_real --config run_teleoperation.yaml --net eno1`；仿真 `python -m deploy.run_teleoperation_mujoco --config run_teleoperation.yaml`（README 命令；机器人侧另需运行 `deploy/teleop/image_server/image_server.py`）。下图按真机脚本 `deploy_real()` 的实际调用画。

```mermaid
sequenceDiagram
    participant Vr as Vision Pro（经 TeleVisionWrapper）
    participant M as deploy/run_teleoperation_real.py
    participant IC as ImageClient 线程 (teleop/image_server)
    participant IK as G1_29_ArmIK (robot_control/robot_arm_ik.py)
    participant Rn as Runner_online_real_dexhand (controllers/controller.py)
    participant Lc as Controller_loco / Controller_squat（ONNX 低层策略）
    participant G as G1 机器人（LowCmd / LowState）
    M->>IC: 启动 receive_process 线程，图像写入共享内存
    M->>M: TeleVisionWrapper(True, tv_img_shape, shm_name)；G1_29_ArmIK()
    M->>Rn: Runner_online_real_dexhand(config, args)
    M->>Rn: damping_state()；move_to_default_pos()
    loop 主循环，模式 SQUAT 或 LOCOMOTION
        alt SQUAT 模式
            M->>Vr: tv_wrapper.get_data() 取头、双腕、双手
            M->>IK: solve_ik(left_wrist, right_wrist, 当前臂关节 q, dq)
            IK-->>M: sol_q（限幅 0.01 rad 写入 target_dof_pos 的手臂部分）
            M->>Rn: run_squat_hand()
            Rn->>Lc: squat_controller.run(cmd_raw, gravity_orientation, omega, qj, dqj, target_dof_pos)
            Rn->>G: pd_control 后 ctrl_dual_hand、send_cmd(low_cmd)
        else LOCOMOTION 模式
            M->>Rn: run_loco()
            Rn->>Lc: loco_controller.run(摇杆 cmd) 内部 loco_low_level_policy.inference
            Rn->>G: pd_control、send_cmd(low_cmd)
        end
        Rn-->>M: transfer_to_loco / transfer_to_squat 触发模式切换
    end
```

图依据：GalaxyGeneralRobotics/OpenWBT@a0154ca，读过 README.md、deploy/run_teleoperation_real.py、deploy/run_teleoperation_mujoco.py、deploy/controllers/controller.py（run_loco、run_squat_hand、Controller_loco.run）、deploy/teleop/robot_control/robot_arm_ik.py（solve_ik 签名）。TeleVisionWrapper、ik_server、handle_controller、policy_unified 的内部未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [OpenWBT 源码归档](../../sources/repos/openwbt.md)（<https://github.com/GalaxyGeneralRobotics/OpenWBT>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：GalaxyGeneralRobotics/OpenWBT@a0154ca，读过 README.md、deploy/run_teleoperation_real.py、deploy/run_teleoperation_mujoco.py、deploy/controllers/controller.py、deploy/teleop/robot_control/robot_arm_ik.py（solve_ik 签名）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [银河通用 官方组织](https://github.com/GalaxyGeneralRobotics)

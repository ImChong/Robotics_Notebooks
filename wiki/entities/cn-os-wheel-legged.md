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
  - ../../sources/repos/wheel-legged.md
summary: "达妙科技 开源项目 wheel-legged（工程与工具）：提供达妙轮足机器人控制与开发资料，用于平衡、移动和执行器系统联调。…"
institutions:
  - dmbots
---

# wheel-legged

## 一句话定义

**wheel-legged** 是 [达妙科技](https://github.com/dmBots) 公开的 **工程与工具** 开源项目：提供达妙轮足机器人控制与开发资料，用于平衡、移动和执行器系统联调。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 达妙科技 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/dmBots |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wheel-legged` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`Core/Src/main.c` 的 `main()`，由 `硬件/打印件版/代码/DM-balance打印件版本/DM-balanceV1/MDK-ARM/CtrlBoard-H7_IMU.uvprojx`（MDK-ARM）编译烧录，上电复位后执行（固件，非命令行程序）

```mermaid
sequenceDiagram
    participant M as Core/Src/main.c main
    participant BM as Devices/BMI088 BMI088_init
    participant CB as Bsp/can_bsp FDCAN1_Config FDCAN2_Config
    participant FR as Core/Src/freertos.c MX_FREERTOS_Init
    participant IN as APP/INS_task INS_task
    participant CM as chassis_move 共享状态
    participant CL as APP/chassisL_task ChassisL_task
    participant VMC as Algorithm/VMC VMC_calc_1_left
    participant DM as Devices/DM_Motor 电机驱动
    participant RX as Bsp/can_bsp FDCAN 接收回调
    participant PS as APP/ps2_task pstwo_task
    participant OB as APP/observe_task Observe_task
    M->>BM: BMI088_init 循环直到成功（SPI2）
    M->>CB: FDCAN1_Config 与 FDCAN2_Config
    M->>FR: MX_FREERTOS_Init 创建任务
    M->>M: osKernelStart 启动调度
    FR->>IN: 创建 INS_TASK（实时优先级）
    FR->>CL: 创建 CHASSISL_TASK（另有 CHASSISR_TASK、OBSERVE_TASK、PS2_TASK）
    loop INS 任务
        IN->>BM: BMI088_Read 读加速度与角速度
        IN->>IN: mahony 滤波更新 INS 四元数与姿态角
        IN->>CM: 收敛后置 INS.ins_flag 为 1
    end
    CL->>CL: 等待 INS.ins_flag 后执行 ChassisL_init（电机 id 6、8、1 进入 MIT 模式）
    loop 左腿控制循环（约 3 ms）
        CL->>CM: chassisL_feedback_update 读取关节角与 INS
        CL->>VMC: VMC_calc_1_left 得到 theta、d_theta 与腿长 L0
        CL->>CL: LQR_K_calc 查增益，计算轮毂力矩 wheel_T 与髋关节力矩 Tp
        CL->>CL: ground_detectionL 离地检测，PID_Calc 得到 F0
        CL->>DM: mit_ctrl 与 mit_ctrl2 经 hfdcan2 下发力矩
    end
    loop 每 10 ms
        PS->>PS: PS2_data_read 读手柄数据
        PS->>CM: PS2_data_process 写 start_flag、v_set、leg_set
    end
    Note over RX: 电机经 CAN 回传反馈帧，触发 HAL_FDCAN_RxFifo0 或 1 回调
    RX->>CM: dm4310_fbdata 或 dm6215_fbdata 解析写入电机状态
    loop 观测
        OB->>OB: xvEstimateKF_Update 卡尔曼融合加速度与速度
    end
```

图依据：达妙 wheel-legged@a6613d4，`硬件/打印件版/代码/DM-balance打印件版本/DM-balanceV1/` 下的 Core/Src/main.c（main 初始化段）、Core/Src/freertos.c（任务创建与任务入口）、User/APP/INS_task.c、chassisL_task.c、ps2_task.c、observe_task.c（主循环段）、User/Bsp/can_bsp.c（FIFO 接收回调）、User/Devices/DM_Motor/dm4310_drv.c（仅 grep 函数签名）；VMC、LQR 与 PID 相关函数、BMI088driver.c 仅从调用端确认。右腿 ChassisR 与左腿对称（经 hfdcan1），未单独画出。另有机加版与群友版固件目录，本图只取打印件版本。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wheel-legged 源码归档](../../sources/repos/wheel-legged.md)（<https://github.com/dmBots/wheel-legged>）

- 源码核对（2026-11-02）：dmBots/wheel-legged@a6613d4，读过 硬件/打印件版/代码/DM-balance打印件版本/DM-balanceV1/ 下 Core/Src/main.c、Core/Src/freertos.c、User/APP/INS_task.c、User/APP/chassisL_task.c、User/APP/ps2_task.c、User/APP/observe_task.c、User/Bsp/can_bsp.c；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [达妙科技 官方组织](https://github.com/dmBots)

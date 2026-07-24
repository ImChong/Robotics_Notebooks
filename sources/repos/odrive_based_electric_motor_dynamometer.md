# Capo01 / ODrive 开源电力测功机

> 来源归档

- **标题：** Odrive based electric motor dynamometer
- **类型：** repo
- **来源：** <https://github.com/Capo01/odrive_based_electric_motor_dynamometer>
- **关联硬件生态：** [ODrive Robotics](https://odriverobotics.com/)
- **入库日期：** 2026-07-24
- **一句话说明：** 基于 ODrive 的开源 **四象限** 无刷电机对拖测功台：吸收电机加载 + 负载传感器测扭矩，可扫效率地图与静态 \(K_t\)。
- **代码：** <https://github.com/Capo01/odrive_based_electric_motor_dynamometer>（已开源；README 标注项目曾暂停维护）
- **沉淀到 wiki：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

## 为什么值得保留

- 补齐 Magtrol 类 **吸收式单向测功** 之外的 **电力对拖 / 再生** 一手实现，贴近人形实验室「一台作负载、一台作 DUT」的常见搭法。
- 明确给出可复现的传感器布局（摆臂 + load cell）与母线回馈思路，便于 Stage 6 台架自学搭建。

## 开源核查（2026-07-24）

| 项 | 状态 |
|----|------|
| 仓库 | 公开 GitHub |
| 内容 | README、测试流程、样例电机数据、台架照片 |
| 许可 | 以仓库根目录声明为准（查阅时需打开 LICENSE） |
| 维护 | README 写明曾因其他事务 **on hold**；仍可作为架构参考 |

## 设计要点（README）

```mermaid
flowchart LR
  DUT["被测电机 DUT"]
  CPL["联轴器"]
  ABS["吸收电机<br/>D6374 级"]
  LC["Load cell + 摆臂"]
  ODRV["ODrive 控制"]
  MCU["辅 MCU<br/>母线 V/I"]

  DUT --> CPL --> ABS
  ABS --> LC
  ODRV --> DUT
  ODRV --> ABS
  MCU --> ODRV
```

| 能力 | 量级 / 说明 |
|------|-------------|
| 功率 / 转速 | ~50–2000 W，0–7500 rpm |
| 峰值制动扭矩 | ~3.5 N·m（可换吸收机适配不同电机） |
| 四象限 | 吸收机可制动或驱动 → 电动/发电工况 |
| 扭矩传感 | 吸收机机座摆臂 + 负载传感器 |
| 电功率 | 分流器 + 母线电压 |
| 可测项 | 效率/损耗地图、静态扭矩 → \(K_t\)、相 R/L（室温与升温）、空载最高速 |
| 能量 | 制动能量回馈至被测侧母线，降低电源需求 |

控制由 ODrive + Python 脚本完成，结果输出文本文件；样例含 D5065 270kv 效率地图拟合图。

## 局限（入库时）

- 扭矩量程偏小关节人形峰值（数十–上百 N·m）不足，需放大传感器与吸收机；架构可迁移。
- 项目维护不活跃；安全互锁、标定溯源需自建。
- 与国标 GB/T 43200 **模组级** 试验项（背隙、总线力矩带宽等）无直接覆盖，仅电机/小功率对拖层。

## 对 wiki 的映射

- [电机测功机（Dynamometer）](../../wiki/concepts/motor-dynamometer.md)
- [电机转矩-电流曲线（TI）](../../wiki/concepts/motor-torque-current-curve.md) — 静态 \(K_t\) 测量路径
- [SimpleFOC](../../wiki/entities/simplefoc.md) — 同属 DIY 驱动/台架生态对照

## 相关一手索引

- [motor_dynamometer_primary_refs.md](../sites/motor_dynamometer_primary_refs.md)

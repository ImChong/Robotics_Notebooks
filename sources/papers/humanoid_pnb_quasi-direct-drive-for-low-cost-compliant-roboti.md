# Quasi-Direct Drive for Low-Cost Compliant Robotic Manipulation

> 来源归档（ingest · Robot Learning Paper Notebooks progress 待深读）

- **标题：** Quasi-Direct Drive for Low-Cost Compliant Robotic Manipulation
- **类型：** paper
- **深读状态：** 待撰写（见 [papers/PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)）
- **计划笔记路径：** `papers/12_Hardware_Design/quasi-direct-drive-for-low-cost-compliant-roboti/quasi-direct-drive-for-low-cost-compliant-roboti.md`
- **分类：** 12_Hardware_Design
- **arXiv：** <https://arxiv.org/abs/1904.03815>
- **项目页：** <https://berkeleyopenarms.github.io/>（[本地归档](../sites/berkeley-open-arms-blue.md)）
- **代码：** <https://github.com/berkeleyopenarms/blue_core>（[本地归档](../repos/blue-core.md)；系统软件部分开源）
- **入库日期：** 2026-06-11（2026-11-21 原文 PDF + `blue_core` 核心源码精读补全）
- **通读：** 已完成
- **一句话说明：** 列入 Paper Notebooks 阅读进度，已完成 `arXiv:1904.03815v2` 全文与 `berkeleyopenarms/blue_core` 核心 ROS 接口文件（`BLDCDriver_node.cpp`、`blue_hardware_interface.cpp`）通读与数据核对。

## 核心摘录（原文 PDF 精读实证）

- **全臂架构与定位**：7-DoF 人尺度准直驱柔顺机械臂 **Blue**（UC Berkeley，Gealy / McKinley / Abbeel 等，ICRA 2019），臂展端额定负载 **2 kg**，量产 >1,500 臂目标终端价 `< $5,000`；实测 7 台原型平均单臂 BOM **$3,328**（含一次性模具 NRE 等约 $160k 工具 + $12k NRE 未摊销）。
- **机械与传动**：全臂采用 **15 mm GT3** 单级同步带减速，减速比 **7.125:1**（16T→114T，机械效率 >95%，同步带刚度约 1.3 kN·m/rad、整机等效刚度约 1.2 kN·m/rad）；肩/肘/腕使用 2-DoF 塑料齿轮差动模块共享双电机载荷，并把肩肘电机后移至近端基座，使肩部重力矩与连杆转动惯量估算各降低约 **30%**。
- **感知、电子与控制**：14-bit 磁编码器 + 12-bit 相电流采样 + 3 轴加速度计 + 温度检测；板载本地电流环运行在 **20 kHz**，上位机经 **1 Mbps RS485** 以 **170 Hz** 轮询读取/下发指令。
- **实测性能**：home repeatability 半径 **3.7 mm**（平均偏差 **2.6 mm**）；力矩滞回 **2.6 N·m**（其中单个带轮模组齿槽转矩 **0.47 N·m**、空载摩擦/滞回共约 **0.89 N·m**）；保守力矩控制带宽 **13.8 Hz**（10 N·m、0.1–60 Hz chirp，对比人类二头肌 2.3 Hz），名义位置控制带宽 **7.5 Hz**。
- **热学与持续工作**：无风扇热导约 **0.3 W/°C**，加风冷升至 **0.93 W/°C**；70 °C 下单电机连续功率约 **40 W**（对应关节约 20 N·m / 0.5 kg）；热容 **1103 W·s/K**，支持 2 min 峰值突发后 7 min 冷却（23% 占空比，2 kg 负载下 >35 N·m 且保留 10 N·m 裕量）；Pick-and-place 中前三个近端关节消耗约 90% 功率。
- **代码架构**：ROS 元包 `berkeleyopenarms/blue_core` 由 `blue_bringup` → `blue_controller_manager` → `blue_controllers` → `blue_hardware_interface`（`BlueHW` 类注册 `joint_state`/`joint_effort`/`joint_position` 接口并调用重力补偿动力学与差动运动学映射）→ `blue_hardware_drivers`（`BLDCDriver_node.cpp` / `BLDCControllerClient`）组成。
- 知识归纳见 wiki 实体页：[paper-notebook-quasi-direct-drive-for-low-cost-compliant-roboti](../../wiki/entities/paper-notebook-quasi-direct-drive-for-low-cost-compliant-roboti.md).

## 对 wiki 的映射

- [paper-notebook-quasi-direct-drive-for-low-cost-compliant-roboti](../../wiki/entities/paper-notebook-quasi-direct-drive-for-low-cost-compliant-roboti.md)
- 分类父节点：[paper-notebook-category-12-hardware-design](../../wiki/overview/paper-notebook-category-12-hardware-design.md)

## 参考来源（原始）

- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/1904.03815>
- 项目页：[berkeley-open-arms-blue.md](../sites/berkeley-open-arms-blue.md)
- 代码：[blue-core.md](../repos/blue-core.md)

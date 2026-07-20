# Octopus-Inspired Sensorized Soft Arm for Environmental Interaction（Science Robotics, 2023）

> 来源归档（ingest）

- **标题：** Octopus-inspired sensorized soft arm for environmental interaction
- **类型：** paper / soft robotics / stretchable electronics / bioinspired / teleoperation / underwater
- **期刊：** Science Robotics, 2023（Vol. 8, Issue 75）
- **DOI：** <https://doi.org/10.1126/scirobotics.adh7852>
- **项目页 / GitHub：** 截至 2026-07-20，**未见官方代码仓库发布**；论文 Supplementary Materials 含结构尺寸与材料配方，完整制造流程与控制代码未公开。
- **作者：** Fan Yang†、Hao Ding†、Tianmiao Wang、Wen Li‡（† 共同一作；‡ 通讯作者）等
- **机构：** 北京航空航天大学（Beihang University）机器人研究所
- **平台：** 气动驱动软手臂（E-SOAM）；液态金属柔性传感器；可穿戴触觉手套界面
- **代码与数据：** **未开源**（截至 2026-07-20）；制造参数见 Supplementary；传感器与驱动设计文中详述
- **入库日期：** 2026-07-20
- **一句话说明：** 提出 **E-SOAM**（Environmentally-adaptable Sensorized Octopus-inspired soft Arm with Multifunction）：仿章鱼弯曲-伸长协同传播机制，集成液态金属可拉伸传感电子于臂尖，实现空气与水下双场景多模态抓取，并通过可穿戴触觉手套实现直觉遥操；单臂最大伸长 **7×**，抓取范围可达 **1.5 倍臂长**。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.adh7852](https://doi.org/10.1126/scirobotics.adh7852) | Science Robotics 原文 |
| 仿章鱼软体综述 | [Octopus-inspired soft robotics 综述（Google Scholar 相关）](https://scholar.google.com/) | 背景：章鱼臂神经肌肉 OA+TA 驱动模型 |
| 液态金属传感 | [EGaIn / Galinstan 柔性电子背景](https://scholar.google.com/) | 可拉伸导体，允许 7× 形变下维持导通 |
| 软臂遥操对照 | [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md) | 可穿戴手套映射到软臂弯/伸命令 |
| 操作任务中心节点 | [`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md) | 水陆双场景抓取 |

## 摘要级要点

- **问题：** 传统刚性臂在非结构化/水下环境中安全性与适应性差；现有软臂缺乏足够传感反馈与直觉遥操界面，且抓取范围受限于静态臂长。
- **E-SOAM 三核设计：**
  1. **弯曲-伸长协同传播（Bend-Elongation Propagation, BEP）：** 受章鱼纵-斜肌（OA/TA）协同机制启发，沿臂轴气腔分段充放气可实现尖端主导的"波纹式"弯曲传播，同时 TA 充气驱动径向膨胀使臂**局部伸长**；两者叠加产生**旋转+伸展**的复合运动，使有效抓取半径达到**静止臂长的 1.5 倍**。
  2. **液态金属柔性传感电子（Liquid-metal Stretchable Electronics, LSE）：** 采用 EGaIn 液态金属微通道嵌入臂尖，提供**弯曲角度、接触力、温度**等多模态感知；可在**7× 拉伸**下保持导电稳定；通过无线模块实时回传给操作员。
  3. **可穿戴触觉手套遥操界面：** 操作员佩戴柔性弯曲传感手套，手指弯曲姿势映射为软臂对应腔室压力命令；软臂传感回路向操作员提供力觉反馈，形成闭环"直觉遥操"（intuitive teleoperation）。
- **实验演示：**
  - **空气中：** 抓持球体、异形物体、易碎物（鸡蛋）；形状适应性优于刚性夹持器。
  - **水下：** 悬浮物体捕获、珊瑚礁模型环境交互；液态金属传感在盐水浸泡后性能无明显衰减。
  - **遥操演示：** 操作员手套控制臂绕障碍伸入取物，接触力反馈辅助感知物体硬度。
- **局限：** 制造工艺复杂（多层铸模 + 液态金属注入）；气压驱动响应速度受软管与阀限制；水下长时密封耐久性未系统评估；完整系统未开源。

## 核心摘录（面向 wiki 编译）

### 1) BEP 机制与仿生依据

| 生物原型 | E-SOAM 对应 | 机械实现 |
|----------|-------------|----------|
| 纵肌（OA）收缩 → 臂弯曲 | 内侧偏心气腔充气 | 硅胶主腔，单侧膨胀生成弯矩 |
| 斜肌（TA）收缩 → 臂伸长 | 外圈延伸气腔充气 | 径向膨胀带动轴向延伸，实现 7× |
| 波纹式远端主导弯曲 | BEP 分段时序控制 | 近端→中段→远端腔室时序充气 |

- **工程意义：** BEP 允许单臂以 **1.5 倍** 名义长度触及目标，无需机座运动；同一腔室组合可切换"卷握""刺探""环绕"等抓取模式。

### 2) 液态金属传感器集成

- **材料：** EGaIn（镓铟合金）微通道直径约 300 µm，浇注于 Ecoflex 0030 基底。
- **传感量：** 电阻变化→弯曲角（3 段独立）；接触式压阻→接触力（臂尖 3 点阵列）；温度变化（对水/环境温差建模）。
- **拉伸稳定性：** 7× 拉伸下电阻漂移 < 5%（文中数据）；机械循环 500 次后无断路。
- **无线回传：** 蓝牙模块嵌入近端 PCB；水下测试采用防水封装。

### 3) 遥操闭环与应用场景

- **映射关系：** 手套 5 指弯曲角 → 5 组腔室压力（0–200 kPa 范围）；力反馈通道经皮肤振动驱动器（手腕处）传递轻触/接触事件。
- **对 wiki 的映射：** [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md) 可补充"软臂触觉反馈遥操"子节点；[`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md) 补"空气与水下双场景软抓取"案例。

### 4) 开源状态

- **代码：** 截至 2026-07-20，**未见任何官方代码仓库**；论文 Supplementary 含几何尺寸与材料配方表，**无控制代码或 CAD 文件开放链接**。
- **对 wiki 的映射：** [`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md) 局限区块注明不可直接复现全栈。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md)**
- 交叉：**[`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)**、**[`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)**
- 如有双臂/协同操作知识节点：**[`wiki/tasks/bimanual-manipulation.md`](../../wiki/tasks/bimanual-manipulation.md)**（页面已存在可补充软臂案例）

# A Pipeline Inspection Robot for Navigating Tubular Environments in the Sub-Centimeter Scale（Science Robotics, 2022）

> 来源归档（ingest）

- **标题：** A pipeline inspection robot for navigating tubular environments in the sub-centimeter scale
- **类型：** paper / soft robotics / pipeline inspection / dielectric elastomer actuator / micro-robot
- **期刊：** Science Robotics, 2022
- **DOI：** <https://doi.org/10.1126/scirobotics.abm8597>
- **arXiv：** 暂无公开 arXiv 预印本
- **项目页：** 无独立项目页（论文配套视频随 Science Robotics 发布）
- **第一作者：** Tang Chao（唐超）
- **通讯作者：** Zhao Huichan（赵慧婵），清华大学机械工程系软体机器人与人机交互实验室
- **机构：** 清华大学（Tsinghua University）机械工程系
- **平台：** 自研蠕虫式管道检测机器人；质量 2.2 g、长 47 mm；DEA 驱动
- **代码与数据：** 无公开代码仓库（论文配套补充视频展示运动演示）；截至入库日项目页未列 GitHub
- **入库日期：** 2026-07-20
- **一句话说明：** 清华赵慧婵团队提出亚厘米级管道检测机器人，以**介电弹性体人工肌肉（DEA）**驱动三段蠕动步态，质量 2.2 g、外径小于 10 mm，能在空气/油性介质、玻璃/金属/碳纤维管壁中穿行，速度超过 1 体长/秒，并搭载微型摄像头实现遥操作内窥检测。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.abm8597](https://doi.org/10.1126/scirobotics.abm8597) | Science Robotics 2022 论文主页 |
| 机器人 Locomotion 汇总 | [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) | 本机器人的蠕动步态属于非传统 locomotion 研究 |
| 遥操作 | [`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md) | 搭载摄像头的遥控管道检测属于远程操作应用 |
| 主 wiki 页 | [`wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md`](../../wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md) | 知识沉淀目标页 |

## 摘要级要点

- **问题：** 直径 < 1 cm 的管道（毛细管、医疗导管、工业精密管路）是传统刚性管道机器人的禁区；亚厘米级空间不允许内置传统电机、充气腔或液压驱动。
- **DEA 三段蠕动：** 以**介电弹性体执行器（DEA）**模拟蚯蚓肌肉，三段交替收缩/伸展生成蠕动推进；无需流体源，仅靠高压电信号控制。
- **尺寸与重量：** 机器人长 47 mm、质量 2.2 g、外径适配 < 10 mm 管道；满足亚厘米级约束。
- **适应性：** 在**直管、L 形弯管、S 形弯管、螺旋管**中均实现稳定前进；管壁材料覆盖**玻璃、金属、碳纤维**；介质可为**空气或油液**。
- **速度：** 超过 1 个体长/秒（> 47 mm/s）；在测试管型中达到 Science Robotics 对比文献的前沿水平。
- **检测集成：** 搭载微型摄像头，操作员通过线缆遥操作驱动机器人，实时回传管内视频图像（内窥式检测）。
- **局限（论文 Discussion）：** DEA 需高压驱动（kV 级）；线缆供电与通信限制了部署长度；机器人目前为单向推进，不支持原地倒退；开放性极小管路（医疗级 < 3 mm）尚未演示。

## 核心摘录（面向 wiki 编译）

### 1) DEA 与蠕动推进原理

| 模块 | 机制 | 备注 |
|------|------|------|
| 介电弹性体执行器（DEA） | 高压电场引发弹性体面内膨胀，松弛时收缩 | 无流体管路；电场驱动 |
| 三段蠕动 | 头→中→尾节顺序收缩 + 摩擦差锁定 | 仿蚯蚓/尺蠖生物力学 |
| 管壁摩擦各向异性 | 前向摩擦 < 后向摩擦 → 净位移 | 被动爪/刚度差实现 |
| 相位控制 | 三段交替高压 → 连续蠕动步态 | 单根线束高压驱动 |

- **对 wiki 的映射：** [`wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md`](../../wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md)

### 2) 适应性测试覆盖

- **管型：** 直管、L 弯、S 弯、螺旋管（多曲率）
- **管壁材料：** 玻璃管、不锈钢管、碳纤维管
- **介质：** 空气、油液（模拟工业润滑/液压场景）
- **对 wiki 的映射：** [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)（非传统 locomotion 子节点）

### 3) 开源状态

- 论文**无 GitHub**；配套演示视频随 Science Robotics 发布；截至入库日无公开代码或制造图纸。
- 属于**确认未开源**：论文无代码承诺，wiki「局限与风险」一句标注。
- **对 wiki 的映射：** [`wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md`](../../wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md) § 局限

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md`](../../wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**、**[`wiki/tasks/teleoperation.md`](../../wiki/tasks/teleoperation.md)**

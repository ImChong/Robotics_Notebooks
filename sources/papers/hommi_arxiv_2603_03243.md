# HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations
- **类型：** paper
- **状态：** RSS 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2603.03243>
  - 项目页：https://hommi-robot.github.io/
- **代码：** https://github.com/xxm19/hommi
- **作者：** Xiaomeng Xu, Jisang Park, Han Zhang, Eric Cousineau, Aditya Bhat, Jose Barreiros, Dian Wang, Jeannette Bohg, Shuran Song
- **机构：** Stanford University; Toyota Research Institute
- **入库日期：** 2026-08-18
- **一句话说明：** HoMMI（RSS 2026）：UMI+第一人称感知无机器人全身移动操作示范；具身无关视觉+放松头动作+扩散 Transformer WBC；代码数据硬件已开源。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 在 UMI 上加 egocentric 感知采集无机器人全身移动操作数据，用手眼跨具身策略+约束感知 WBC 弥合人机形态差。
- **对 wiki 的映射：**
  - [wiki/entities/paper-hommi.md](../../wiki/entities/paper-hommi.md)

### 方法与结果（归纳）

- **方法：** 具身无关 3D 视觉表征；放松 look-at-point 头动作；扩散 Transformer 全身控制满足机器人约束。
- **评测：** 长时程双臂移动操作：导航、双手协调、主动凝视；无机器人 teleop 数据训练。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-hommi.md`

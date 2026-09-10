# TANGO: Humanoid Navigation in Cluttered Environments with a Whole-Body Vision-Language-Action Model（arXiv:2609.09158）

> 来源归档（ingest）

- **标题：** TANGO: Humanoid Navigation in Cluttered Environments with a Whole-Body Vision-Language-Action Model
- **简称：** TANGO
- **类型：** paper / humanoid / vla / vision-language-navigation / whole-body
- **arXiv：** <https://arxiv.org/abs/2609.09158>
- **PDF：** <https://arxiv.org/pdf/2609.09158>
- **项目页：** <https://tango-vla.github.io/tango-vla.github.io> — 归档见 [`sources/sites/tango-vla.md`](../sites/tango-vla.md)
- **会议：** CoRL 2026
- **机构：** 北京大学、加州大学伯克利分校（Tomizuka）、普林斯顿大学 / Google DeepMind（Dhruv Shah）等（以论文作者列表为准）
- **入库日期：** 2026-09-10
- **一句话说明：** 首个全身 VLA 语言导航：仿真合成碰撞自由穿越行为（路径规划→全身运动生成→障碍编辑→RL tracking）监督 29-DoF 关节动作；G1 零样本真机 cluttered 场景。

## 开源状态（步骤 2.5，2026-09-10）

- **结论：** **截至入库日未开源** — 项目页有 PDF、视频与 BibTeX，**无 GitHub / 权重链接**。

## 核心摘录（面向 wiki 编译）

### 摘录 1：问题与输出

- 室内杂乱环境人形导航不是 2D 路径规划：需要 **连续几何感知全身适应**（摆臂、躯干、步态调制）。
- 输入自然语言 + 第一人称 RGB；直接预测 **29-DoF 关节空间动作** 供下游全身控制。

**对 wiki 的映射：** [paper-tango-vla](../../wiki/entities/paper-tango-vla.md)

### 摘录 2：仿真数据合成管线

- 全局路径规划 → 运动学全身运动生成 → **障碍感知运动编辑** → **RL tracking** 得到动力学可行监督。
- 全程仿真训练；**零样本** 部署 Unitree G1，无真机导航数据。

**对 wiki 的映射：** [paper-tango-vla](../../wiki/entities/paper-tango-vla.md)

## 当前提炼状态

- [x] 项目页核查（2026-09-10）
- [x] wiki 实体页已建

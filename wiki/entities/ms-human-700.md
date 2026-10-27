---
type: entity
project_id: ms-human-700
tags: [paper, biomechanics, human-motion, musculoskeletal-model, reinforcement-learning, mujoco, tsinghua]
status: complete
updated: 2026-10-27
arxiv: "2312.05473"
venue: "ICRA 2024"
project: https://lnsgroup.cc/research/MS-Human-700
code: https://github.com/LNSGroup/MS-Human-700
related:
  - ./embodied-human-simulation.md
  - ./mujoco.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/ms_human_700_arxiv_2312_05473.md
  - ../../sources/sites/ms-human-700.md
  - ../../sources/repos/lnsgroup-ms-human-700.md
  - ../../sources/repos/lnsgroup-msgym.md
summary: "全身人体肌骨 MuJoCo 模型，含 90 个身体节段、206 个关节和 700 条肌腱单元；以层级低维表示和强化学习控制高维肌肉系统，并提供开放模型、任务环境与训练代码。"
---

# MS-Human-700：全身人体肌骨模型

**一句话定义：** MS-Human-700 是用于 MuJoCo 全身动力学与运动控制研究的人体肌骨模型，以 700 条肌腱单元驱动身体运动，并配套公开模型资产和强化学习环境。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ICRA | IEEE International Conference on Robotics and Automation | 原始论文发表会议 |
| RL | Reinforcement Learning | 学习肌骨运动控制策略 |
| MJCF | MuJoCo XML Format | 模型 XML 与仿真场景描述格式 |
| SAC | Soft Actor-Critic | msgym 训练脚本使用的策略优化算法 |
| DoF | Degrees of Freedom | 关节或系统的自由度 |

## 为什么重要

常见人形机器人模型以电机和刚性连杆为核心，而 MS-Human-700 显式建模肌肉—肌腱驱动，适合研究人体运动控制、生物力学仿真，以及用人体数字模型辅助交互机器人设计。700 个肌肉执行器也使它成为高维、过驱动控制问题的研究平台。

## 核心原理

原始工作针对「数百条肌肉如何协调产生全身运动」这一难题，将低维动作表示与层级强化学习结合，降低策略直接搜索所有肌肉激励的难度。论文报告以真实人体运动数据验证仿真控制。

仓库提供三个用途不同的模型变体：

| 模型 | 主要用途 | 身体节段 | 关节 | 肌肉执行器 |
|------|----------|----------:|------:|-----------:|
| MS-Human-700.xml | 全身运动、上下台阶、扰动恢复 | 90（优化结构约 80） | 206（约束后 85） | 700 |
| MS-Human-700-Locomotion.xml | 简化上身的下肢运动 | 80 | 36 | 100 |
| MS-Human-700-Manipulation.xml | 右臂与手部操作 | 127 | 42 | 81 |

这些配置不能互换：若研究全身肌肉协同应使用 700 肌肉主模型；腿部与操作变体减少了执行器规模并改变了模型细节。

## 工程实践与开源状态

- **模型查看：** 用 MuJoCo simulate 打开仓库中的 MS-Human-700 XML 文件。
- **环境安装：** 按 msgym README 指引递归克隆子模块，再安装 Python 包。
- **训练/评估：** DynSyn-SAC 的 train.py 与 eval.py 读取任务配置；仓库 README 提供全身、腿部和操作环境入口。
- **开源状态：** 模型仓库和环境仓库均公开，采用 Apache-2.0；msgym README 指向 Releases 中的训练 checkpoint。
- **解释边界：** 公开实现提供可运行的 MuJoCo 控制环境，但仿真结果不能直接等同于真实人体测量或临床有效性。

## 局限与风险

- 肌肉模型的生理参数、接触建模和控制策略会影响运动结果；使用者应核对任务配置和验证范围。
- 全身主模型、下肢模型和操作模型的身体结构及执行器规模不同，报告实验时应明确模型 XML。
- 项目不是纯动作数据集；参考动作是控制任务输入之一，模型和环境才是主要发布内容。
- BAAI Hugging Face 打包页说明不包含训练好的策略 checkpoint；权重可用性应以对应仓库 Releases 为准。

## 结论

**总判：** MS-Human-700 的价值在于把详细肌骨动力学和高维运动控制放进可复现的 MuJoCo/Gymnasium 工作流，适合肌骨控制与人机交互仿真研究。

- 全身肌肉协同研究优先查看 700 肌肉主模型及原始 ICRA 论文。
- 低维策略表示是降低数百维肌肉动作搜索难度的关键思路。
- 运动和操作任务入口在 msgym；后续 DynSyn-SAC 实现不应直接视为原始论文算法的同义版本。
- 解读结果时需区分人体运动轨迹跟踪与真实生理响应验证。

## 关联页面

- [具身人体仿真](./embodied-human-simulation.md)：MS-Human-700 是数字人体与交互机器人耦合研究的一项肌骨模型基础。
- [MuJoCo](./mujoco.md)：模型使用的物理仿真与 MJCF 平台。
- [强化学习](../methods/reinforcement-learning.md)：理解层级策略训练和 SAC 的算法背景。

## 推荐继续阅读

- [ICRA 2024 原论文](https://arxiv.org/abs/2312.05473)
- [模型与 MJCF 资产](https://github.com/LNSGroup/MS-Human-700)
- [Gymnasium 环境和训练代码](https://github.com/LNSGroup/msgym)

## 参考来源

- [论文来源归档](../../sources/papers/ms_human_700_arxiv_2312_05473.md)
- [官方项目页归档](../../sources/sites/ms-human-700.md)
- [模型仓库归档](../../sources/repos/lnsgroup-ms-human-700.md)
- [训练环境归档](../../sources/repos/lnsgroup-msgym.md)

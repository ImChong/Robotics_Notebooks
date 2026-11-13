# embodied-first-exam.ai（项目页）

- **标题：** Embodied First Exam
- **类型：** site / project-page
- **URL：** <https://www.embodied-first-exam.ai/>
- **代码：** <https://github.com/mll-lab-nu/embodied-first-exam> — [仓库归档](../repos/embodied-first-exam.md)
- **评测框架：** [Harbor](https://github.com/laude-institute/harbor)
- **发布：** 2026-10-19（日期见官方 GitHub README）
- **入库日期：** 2026-10-22
- **复核日期：** 2026-11-13
- **论文 PDF：** <https://arxiv.org/pdf/2611.12177>
- **论文 HTML：** <https://arxiv.org/html/2611.12177v1>
- **开放轨迹：** <https://huggingface.co/datasets/MLL-Lab/embodied-exam-data>

## 一句话摘要

面向代码代理的具身任务评测：在多个机器人 benchmark 的冻结任务场景中，代理编写程序、运行并观察结果，最后由独立沙箱验证；支持通过 Harbor 运行。

## 核验要点

- 论文测试的是 coding agent 自由选择直接控制、写控制器或仿真训练策略的具身能力，并不是新的单一 VLA 权重模型。
- 三种测试设置：Standard 为机器人传感器与控制 API、仿真在 agent 思考时暂停；Realtime 继续推进；Privileged 允许全状态与无限重置。三种设置不同的成功率不应直接混算。
- 官方代码 GitHub 为可运行开源仓库，存在上游仿真器/资产的额外许可证和准备要求。

## 对 wiki 的映射

- 实体页：[Embodied First Exam](../../wiki/entities/embodied-first-exam.md)
- 代码仓：[Embodied First Exam](../repos/embodied-first-exam.md)

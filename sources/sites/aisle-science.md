# AISLE / AI Scientists' Last Exam — 官方项目主页

- **类型：** site（官方项目主页、任务目录及公开排行榜）
- **项目页：** <https://aisle.science/>
- **论文 PDF：** <https://github.com/MasterAI-EAM/ai-scientists-last-exam/blob/main/paper/AI_Scientists_Last_Exam.pdf>
- **代码：** <https://github.com/MasterAI-EAM/ai-scientists-last-exam>
- **排行榜：** <https://aisle.science/leaderboard.html>
- **任务入口：** <https://aisle.science/tasks.html>
- **核查日期：** 2026-11-09
- **一句话说明：** 以真实/模拟实验后的可验证结果为核心的科学智能体评测项目，官方网页列出 colour matching、约束优化、误差恢复、光谱与 pKa 等类别。

## 核查结论

- **开放状态：已开源评测任务与框架代码**。GitHub 有 Apache-2.0 许可证、Harbor 任务定义、环境、验证器及 wetlab 子框架；这不等于硬件控制软件、实验室原始数据、全部模型轨迹和实验设施完全开放。
- 官方站点含 **动态 leaderboard**；不同筛选项（wet/dry、类别、模型）对应不同样本覆盖。分数可能更新，归档不锁定单次快照为论文结果。
- 站点展示物理实验内容，仓库的当前公开 benchmark 任务也包含由模拟模型产生的观测。不能把所有 benchmark 分数解读为真实湿实验成绩。
- 官方 README 明确 **wetlab/VALIDATION.md** 是验证范围边界：模拟实验验证不等价于真实设备端到端验证。

## 对 wiki 的映射

- [AISLE 主实体页](../../wiki/entities/paper-ai-scientists-last-exam.md)
- [论文来源](../papers/ai_scientists_last_exam_2026.md)
- [代码仓库](../repos/ai-scientists-last-exam.md)

# LLM 引导平面移动操作（arXiv:2610.40248）

> 来源归档（paper）

- **标题：** LLM-guided Planar Loco-Manipulation for Quadruped Robots
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40248>
- **PDF：** <https://arxiv.org/pdf/2610.40248>
- **作者：** Jetti, Georges, Khayyat, Michael, Sleiman, Jean-Pierre, Arrigoni, Stefano, Braghin, Francesco
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 大语言模型引导的四足平面移动操作。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 障碍物可移动时，机器人需要联合选择操作技能、目标物体和几何作用点，而非只规划无碰撞路径。
3. **周报所述创新点：** LoRA 微调的语言任务规划器将语言指令和结构化场景转换为可变长度参数化动作；底层端到端全身推动策略在仿真中达到 93.3% 成功率，并在实机 45 次实验中完成 34 次端到端任务。
4. **arXiv 摘要（英文节选）：** Robotic navigation typically assumes a collision-free route to the goal exists; in cluttered environments with movable obstacles, however, reaching the goal may require physical interaction. Classical solutions encode these behaviors through geometric rules and task-specific logic, while learning-based approaches often constrain the decision space to low-dimensional motion commands or predefined action sets. We instead consider a broader decision space, jointly selecting the skill, relevant object, and geometric target.

**对 wiki 的映射**

- [paper-llm-guided-planar-loco-manipulation-quadruped](../../wiki/entities/paper-llm-guided-planar-loco-manipulation-quadruped.md)

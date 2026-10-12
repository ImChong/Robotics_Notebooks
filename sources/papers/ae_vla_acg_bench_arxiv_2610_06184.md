# AE-VLA / ACG-Bench 论文来源归档

## 元数据

- **题目：** Arm-wise Compositional Generalization in Dual-Arm Vision-Language-Action Models
- **方法 / 基准：** AE-VLA；ACG-Bench
- **作者：** Zaibin Zhang, Binghao Ran, Yuhan Wu, Zhongbo Zhang, Yifan Wang, Junwei Jiang, Junlan Xiao, Wangcheng Shi, Li Kang, Yiran Qin, Zhenfei Yin, Lijun Wang, Huchuan Lu
- **单位：** arXiv 摘要页与 HTML 均未列出，未核实
- **预印本：** https://arxiv.org/abs/2610.06184
- **HTML：** https://arxiv.org/html/2610.06184v1
- **版本：** v1，2026-10-05，cs.RO，备注 “Technical Report”
- **项目页 / 代码：** 均未列出（核查日期 2026-10-12）。

## 论文要解决的问题

双臂机器人可能已分别学会多种单臂技能，但新任务要求重排技能顺序、同步左右臂或跨任务重组时，端到端 VLA 往往不能把已有技能按“臂”重新组合。论文把这种能力定义为 arm-wise compositional generalization，并给出基准与结构方案。

## 方法与训练信息

- **ACG-Bench**：基于 ManiSkill，沿用 RoboTwin 2.0 任务与专家程序管线；8 个任务族（6 个源族：Stack Bowls、Stack Cubes、Push Cubes、Burger & Fries、Can in Basket、Object in Cabinet；2 个留出跨任务族：Cube in Bowl、Bowl in Cabinet），23 组任务条件（6 个域内、17 个未见：4 Reorder、6 Sync、4 Sync+Reorder、3 Cross-task）。训练为 600 条源示范（每源任务 100 条）合并的多任务集；每条件评测 100 回合（共 2,300 回合），评分用约束合规成功率 CCSR（需目标 + 里程碑 + 顺序 + 时序）。
- 全部方法基于 π0.5：Single（共享模型控制双臂）、Dual（两个独立单臂策略）、MA-VLA（Arm Shuffle 增广，由 π0 迁到 π0.5）。
- 结构组件：Token Group（每臂一组 H=50 个动作 token，共享一个动作专家，用固定 token 区间识别手臂）；SkillLoRA（10 个 rank-4 LoRA 按技能类型索引，α=16，每臂路由器硬选，路由损失权重 0.1，覆盖 18 个动作专家块）；Arm-wise Attention（AWA，阻断跨臂的直接动作注意力，各动作块下三角，保留全局上下文）；AE-VLA = 三者组合。
- 训练：30,000 步，批 32，2 张 A800，AdamW，峰值 lr 5e-5，流匹配推理 10 步 Euler。

## 论文报告的结果

- 仿真泛化（17 个未见条件）CCSR：Single 2.94，MA-VLA 3.06，Dual 5.53，AE-VLA 21.53；分组：Reorder 26.50，Sync+Reorder 22.00，Cross-task 40.00，纯 Sync 仅 8.67。域内：Single 37.00，AE-VLA 27.17。
- 消融（泛化 CCSR）：Token Group 3.82，+SkillLoRA 6.00，+AWA 5.82，两者都加 21.53；加 AWA 会使域内下降（Token Group 45.00 → 25.00）。
- 真机：两台 6-DoF SO101，1 全局 + 2 腕部相机，Stack Bowls / Stack Cubes / Push Cubes 各 50 条 LeRobot 遥操作示范；3 个域内 + 5 个未见条件，各 20 次。未见平均：Single 2.00%，MA-VLA 6.00%，Dual 10.00%，AE-VLA 39.00%；Stack Bowls/Sync AE-VLA 12/20，所有基线 0/20；域内平均 Dual 70.00%，AE-VLA 60.00%。

## 复现与解释边界

- 评测执行层：每臂提示由外部给定，不含高层规划；论文自述同步与域内性能仍弱。
- SkillLoRA 的收益混杂了路由监督与额外参数，AWA 同时改变跨臂与块内注意力，未做参数匹配或打乱标签控制；错误技能标签未测。
- Dual 策略在两个 Sync 条件的碰撞成功被人工记 0；TwinVLA 因预训练不同未纳入；SkillVLA 官方实现不可用。真机每条件仅 20 次。
- 未声明代码 / 数据发布。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-ae-vla-arm-wise-compositional.md
- 任务入口：wiki/tasks/bimanual-manipulation.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.06184>
- arXiv HTML：<https://arxiv.org/html/2610.06184v1>

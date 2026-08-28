# R^3: Training Robots to Reason in Natural Language via Reinforcement Learning

> 来源归档（ingest）

- **标题：** R^3: Training Robots to Reason in Natural Language via Reinforcement Learning
- **短名：** R³
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.26053>
- **PDF：** <https://arxiv.org/pdf/2608.26053>
- **项目页：** <https://robotic-reasoner.github.io/>
- **代码：** 项目页写明 Coming Soon
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 把自由形式自然语言推理训成低层操作策略可调用的测试时计算接口。

## 开源状态（步骤 2.5）

- **待发布**：项目页按钮为 **Code (Coming Soon)**。截至入库日无 GitHub 训练仓。

## 核心摘录（面向 wiki 编译）

### 摘录 1：两阶段后训练

- 高层 VLM 输出短时程自然语言指令，固定语言条件低层策略执行。
- Stage I：在专家推理轨迹上 mid-train，初始化推理风格（杂货打包可跳过）。
- Stage II：用离线动作数据做单步量表奖励 RL（Dr.GRPO）；Language Table 用 VLM judge，打包任务用有限集合字符串匹配。
- 与把结构化 CoT 仅当辅助监督不同，R³ 直接训练自由形式语言推理作为动作指导。

**对 wiki 的映射：** [paper-r3-robotic-reasoner](../../wiki/entities/paper-r3-robotic-reasoner.md)、[VLA](../../wiki/methods/vla.md)

### 摘录 2：评测

- Language Table：14 个长时程积木排列；R³ 在 OOD held-out 上显著优于仅指令模仿。
- 仿真双臂杂货打包：12 个 held-out 任务，R³（仅 RL）成功率 **47.9%** vs 指令模仿 **38.0%**。

**对 wiki 的映射：** [manipulation](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-r3-robotic-reasoner.md`](../../wiki/entities/paper-r3-robotic-reasoner.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链

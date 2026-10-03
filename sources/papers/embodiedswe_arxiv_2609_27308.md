# EmbodiedSWE: Coding Agents for Long-Horizon Dexterous Robotics

- **标题：** EmbodiedSWE: Coding Agents for Long-Horizon Dexterous Robotics
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.27308>
- **项目页：** <https://embodiedswe.github.io/>
- **代码：** <https://github.com/EmbodiedSWE/EmbodiedSWE>
- **资产数据集：** <https://huggingface.co/datasets/EmbodiedSWE/robobench-assets>
- **作者：** Haoxiang You, Zeyu Shen, Yilang Liu, Zhicheng Zheng, Lihan Zha, Kashu Yamazaki, Mingtong Zhang, Suning Huang, Jiankai Sun, Qianzhong Chen, Lucy He, Kaiyuan Liu, Haoran Chang, Katerina Fragkiadaki, Dhruv Shah, Mac Schwager, Peter Henderson, Ian Abraham, Canwen Xu
- **机构：** ByteDance Seed、Yale University、Princeton University、Carnegie Mellon University、Stanford University、University of California, Los Angeles、University of Washington
- **发表日期：** 2026-09-23
- **入库日期：** 2026-10-03
- **一句话说明：** 提出长时程灵巧机器人任务 benchmark，并用 coding agent 编写可验证程序，再把成功解扩增成 VLA 训练示范。
- **对 wiki 的映射：** [论文实体页](../../wiki/entities/paper-embodiedswe.md)

## 核心摘录

1. **问题：** 遥操作演示成本高，人体视频难直接对应机器人 embodiment；现有 coding-agent 机器人任务多集中导航或简单抓放，长时程、接触丰富任务缺统一评测。
2. **EmbodiedSWE-Bench：** 28 个日常任务，包含装配、整理/打包、谜题、可变形物体与液体、切割、移动操作；任务最长约 30 分钟。论文评测环境包含 Franka、UFactory xArm7、Kinova Gen3、双臂 Franka 与 Unitree G1。
3. **Agent 作为 solver：** coding agent 在仿真中检查状态、执行候选程序、观察结果并迭代，交付入口为 Python `solve(env)`；评估在隔离环境中运行，并由新容器内的 hidden grader 离线打分。
4. **主要结果：** 项目页报告 GPT-6 Astra 在标准 coding harness 上达到 82% success rate；较难任务仍需多轮交互和较高推理成本，策略通常是按具体任务实例定制。
5. **EmbodiedSWE-Gen：** 对已验证解按场景、策略、阶段、动力学与视觉因素分层做多样化，扩展为轨迹数据集；示范数量增加时 SmolVLA 的任务成功率上升，agent 辅助多样化也提高留出变体上的表现。
6. **Sim-to-real：** 论文报告用 500 条 coding-agent 仿真示范微调的 VLA 完成了真实机器人上的四阶段灯具拆解任务；这是单个任务的迁移结果。
7. **开源与许可：** 官方代码仓库与仿真资产数据集均公开。代码及作者制作资产按 Apache-2.0；第三方资产保留上游许可，部分素材为 CC BY-NC / CC BY-NC-SA。

## 相关材料

- [项目页](../sites/embodiedswe-github-io.md)
- [代码仓库](../repos/embodiedswe.md)
- [论文 PDF](https://arxiv.org/pdf/2609.27308)
- [Hugging Face 仿真资产](https://huggingface.co/datasets/EmbodiedSWE/robobench-assets)

## 对 wiki 的映射

- 论文实体页：[EmbodiedSWE](../../wiki/entities/paper-embodiedswe.md)
- 相关研究：[Agentic Coding Agent](../../wiki/entities/paper-agentic-coding-manipulation.md)

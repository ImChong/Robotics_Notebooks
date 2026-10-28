# 自我改进智能体与具身终身学习：论文组

> 来源归档（ingest）；以下为论文摘要和官方项目页的结构化摘录，不替代全文复现。

- **类型：** paper / survey / robot-learning / lifelong-learning
- **入库日期：** 2026-10-28
- **对 wiki 的映射：** [自我改进与具身终身学习对比](../../wiki/comparisons/self-improving-agents-embodied-lifelong-learning.md)

## 论文摘录

### 1. Fang et al.，A Comprehensive Survey of Self-Evolving AI Agents
- **论文：** <https://arxiv.org/abs/2508.07407>（2025，v2）
- **要点：** 用系统输入、智能体系统、环境、优化器四部分描述反馈闭环；按演化对象梳理方法，并讨论评估与安全。
- **资料库：** <https://github.com/EvoAgentX/Awesome-Self-Evolving-Agents>（论文列表，不是算法实现）。

### 2. Ren et al.，Self-Improvements in Modern Agentic Systems
- **论文：** <https://arxiv.org/abs/2607.13104>（2026，v1）
- **要点：** 智能体由基础模型和提示词、记忆、工具、控制逻辑组成；自我改进按模型参数或外围结构的更新目标及信号分类。
- **官方站点：** <https://selfimproving-agent.github.io/>；[站点归档](../sites/selfimproving-agent-survey.md)。站点链接的仓库是论文清单，并非完整训练实现。

### 3. Zhang et al.，Darwin Gödel Machine
- **论文：** <https://arxiv.org/abs/2505.22954>（2025；v3 2026）
- **要点：** 从智能体档案中采样、修改代码、在编程基准上评估，并保留多条探索路径。论文摘要报告 SWE-bench 20.0% → 50.0%，Polyglot 14.2% → 30.7%；仅对应论文实验设置。
- **官方实现：** <https://github.com/jennyzzt/dgm>；[仓库归档](../repos/darwin-godel-machine.md)。论文采用沙箱和人工监督；经验评估不是收益的数学证明。

### 4. Bousmalis et al.，RoboCat
- **论文：** <https://arxiv.org/abs/2306.11706>（2023，v2）
- **项目介绍：** <https://deepmind.google/blog/robocat-a-self-improving-robotic-agent/>；[站点归档](../sites/robocat-deepmind.md)。截至核查日该页面未列官方代码或数据链接，开放状态不能由此断言。
- **要点：** 多任务、多本体视觉目标条件策略；用新任务的人类示范微调，再收集策略生成的轨迹，与原数据合并训练下一代通才。适应新任务需要示范，不等于完全无人参与。

### 5. Yang et al.，RISE
- **论文：** <https://arxiv.org/abs/2602.11075>（2026，v2）；[已有论文归档](sun_awesome_wm_2602_11075_rise-self-improving-robot-policy-with-co.md)；[已有实体](../../wiki/entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md)。
- **项目：** <https://opendrivelab.com/RISE/>；[官方代码](https://github.com/OpenDriveLab/RISE)；仓库 README 声称训练代码与预训练动力学模型已发布。想象空间中以动力学模型预测、进度价值模型评估并更新策略；真机表现仍须单独评测。

### 6. Luo et al.，SERL
- **论文：** <https://arxiv.org/abs/2401.16013>（ICRA 2024）；[已有来源](rcl_awesome_wam_2401_16013_serl-a-software-suite-for-sample-efficie.md)；[已有实体](../../wiki/entities/paper-rcl-2401-16013-serl-a-software-suite-for-sample-efficient-robot.md)。
- **项目：** <https://serl-robot.github.io/>；[算法代码](https://github.com/rail-berkeley/serl)及[控制器](https://github.com/rail-berkeley/serl_franka_controllers)。提供 off-policy RL、奖励、复位和机器人控制组件；论文的每策略 25–50 分钟均值对应特定任务和配置，不能直接推广到任意机器人。

### 7. Gao et al.，Arcadia
- **论文：** <https://arxiv.org/abs/2512.00076>；[已有来源](arcadia_arxiv_2512_00076.md)；[已有实体](../../wiki/entities/paper-arcadia.md)。
- **要点：** 将自主采集、生成场景、共享导航/操作表示和 sim-from-real 反馈连接为生命周期；[EmbodiedKit](../repos/embodiedkit.md) 已有归档对部分开放边界作了核查。

### 8. Yin et al.，AllDayNav
- **论文：** <https://arxiv.org/abs/2606.10927>（2026，v1）
- **项目：** <https://bagh2178.github.io/AllDayNav/>；[站点归档](../sites/alldaynav.md)。截至 2026-10-28 项目页未列官方代码或数据下载链接，不据此断言未开源。
- **要点：** 多模态记忆累积视觉关键帧和语义信息，生成任务、检索目标并配合 CQL 更新导航策略；实验涵盖仿真和真机。项目页的接近 100% 是特定仿真测试场景结果，并非所有场景的保证。

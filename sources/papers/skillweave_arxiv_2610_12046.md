# SkillWeave 论文来源归档

## 元数据

- **题目：** SkillWeave: Weaving Heterogeneous Demonstrations into Long-Horizon Manipulation Skills
- **作者：** Ryosei Tamura, Xiaoxiang Dong, Uksang Yoo, Yuemin Mao, Romina Mir, Jonathan Francis, Jeffrey Ichnowski
- **单位：** 据 arXiv HTML 脚注：卡内基梅隆大学机器人研究所（CMU RI）、庆应义塾大学计算机系、加州大学伯克利分校工程学院、博世人工智能中心（匹兹堡）；脚注未逐作者对应。
- **预印本：** https://arxiv.org/abs/2610.12046
- **HTML：** https://arxiv.org/html/2610.12046v1
- **版本：** v1，2026-10-08，cs.RO；备注：IROS 2026 Workshop on Haptics and Dexterous Manipulation
- **项目页：** http://skillweave-authors.github.io （归档见 sources/sites/skillweave-project.md）
- **代码：** 摘要称项目页提供视频与代码，但项目页实际标注 “Coming Soon”；截至 2026-10-12 无仓库。

## 论文要解决的问题

灵巧长程操作既要长序列，又要接触丰富的精细动作，示范难采。远距离到达/搬运适合遥操作，接触密集的灵巧子技能（拧螺母、开壶盖按钮、手内旋转）更适合动觉示教（kinesthetic teaching，人直接带动机器人）。但动觉示教画面里有示教者的手/身体，造成视觉域偏差；独立训练的子任务策略在交接处又会漂移。

## 方法与训练信息

1. **异构示范：** 平台为 7-DoF xArm7 + 16-DoF LEAP Hand，第三视角 RGB 相机和掌心朝向的腕部相机。每个子任务 100 条示范，遥操作用 6-DoF SpaceMouse 控臂、MediaPipe 手部跟踪经关节角映射控手；动觉示教时臂进入拖动示教模式、操作者在柔顺控制下摆手指。记录 30 Hz 双相机与本体感知（6D 末端位姿、7D 臂关节、16D 手关节），物体位置随机化。
2. **物体掩码条件扩散策略：** 每个相机一个 ResNet-18 编码器，1D U-Net 去噪动作块，本体感知拼接视觉特征；用物体掩码池化出查询向量做注意力池化（物体不可见时用均值特征作查询）。损失 L = L_DP + lambda * L_attn，注意力图由归一化物体掩码监督（不含物体的视角排除；lambda 原文未报告）。训练掩码由 SAM 3 离线生成，部署时每视角一个轻量 U-Net 预测掩码，无需在线分割或图像修补。
3. **后继感知终端引导（successor-aware terminal steering）：** 在前驱策略终点附近采样 L 个动作块，用预测终点到后继策略起始状态集（取后继 N 条示范的前 H_s 个状态）的 K 近邻平均距离 d(s) 打分，执行得分最小者；d < epsilon 或达最大引导步数时停止。腕位姿、臂关节、手关节先归一化。L、K、N、H_s、epsilon、最大步数原文未报告。无需过渡示范或过渡策略，也不修改后继策略。
4. **三个真机任务：** Cube Reorientation（遥操作到达抓取，动觉手内旋转，成功判据旋转至少 90 度）、Nut Removal & Storage（遥操作取螺母，动觉拧下，遥操作放入容器）、Kettle Preparation（遥操作抓壶，动觉按钮开盖，遥操作放壶）。每个评测 20 次试验。

## 论文报告的结果

- 子任务成功率（遥操作 / 原始动觉 / KineDex / 掩码动觉）：手内旋转 10% / 30% / 0% / 70%；拧螺母 5% / 0% / 60% / 75%；按钮开盖 0% / 0% / 0% / 50%；遥操作的到达/搬运类（抓取重定向 70%、取螺母 85%、放螺母 30%、抓壶 80%、放壶 55%）。文中称遥操作臂级子任务平均 64%，掩码动觉灵巧子任务平均 65%（原文另有“遥操作在灵巧子任务 5%”的说法，与表格不完全吻合）。
- 端到端成功率 S（S*=各子任务成功率乘积；组合效率 C=S/S*）：SkillWeave 平均 S 27%、S* 30%、C 87%（Cube 45%/49%/92%，Nut 15%/19%/79%，Kettle 20%/22%/91%）；去掉终端引导平均 S 5%、C 14%；纯遥操作、原始动觉、KineDex 三个基线端到端均为 0%。
- 原文表 II(b) 与正文/表 III 对“有/无引导”的数字标注互换，本归档按正文与表 III 取值。

## 复现与解释边界

- 仅三个真机任务、单一机器人、每项 20 次试验，端到端 27% 绝对值不高；论文无单独局限小节，作者列出的后续方向包括自动分解任务与选择示范模态、减少对显式物体掩码的依赖、扩展到更大技能库与新物体。
- 终端引导的关键超参未披露，且没有可运行代码，难以复现。
- 作者仅为匿名项目页形态（“Under review”），归档以 arXiv 为准。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-skillweave.md
- 任务入口：wiki/tasks/manipulation.md
- 周报汇总：wiki/overview/frontier-manipulation-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.12046>
- arXiv HTML：<https://arxiv.org/html/2610.12046v1>
- 项目页：<http://skillweave-authors.github.io>

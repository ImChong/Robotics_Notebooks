# Cartesian Hand v1（General Robotics Lab 项目页）

> 来源归档（site）

- **标题：** The Cartesian Hand: In-Hand Manipulation with All-Linear Fingers
- **类型：** site
- **链接：** https://generalroboticslab.com/cartesian_handv1
- **机构：** 杜克大学（Duke）General Robotics Lab（Boyuan Chen）
- **关联论文：** [arXiv:2609.25696](https://arxiv.org/abs/2609.25696)
- **入库日期：** 2026-09-28
- **一句话说明：** Duke GRL 官方项目页：7-DoF 全线性双平行夹爪 + 四平移指尖的手内操作末端；35 类实验室/制造/家用铰接物体 demo；固定基座臂与人形双臂实验室操作迁移。

## 项目页核查（步骤 2.5，2026-09-28）

- **代码：** https://github.com/generalroboticslab/Cartesian_Hand（**已开源**，Apache-2.0；README 链到本页）
- **开放程度：** 控制栈 + MuJoCo/STEP 已发布；`cartesian_hand/tasks/` 自述为转录、**未在全部 35 物体上重跑**；完整制造包以仓库后续提交为准
- **项目页：** React SPA，正文与 arXiv 摘要一致；详细安装/命令以 GitHub 为准
- **联系：** info@generalroboticslab.com（以站点为准）

## 机制摘要（与 arXiv 一致）

- **7 个棱柱关节：** $q_0,q_4$ 上下平行夹爪开闭；$q_1,q_2,q_5,q_6$ 四指尖沿固定轴平移；$q_3$ 上下夹爪间距。
- **驱动：** Feetech STS3915 舵机 + 齿条–齿轮；指尖行程约 65 mm，夹爪/间距轴约 52 mm；整机约 **850 g**，约 **166×100×76 mm**（闭合）。
- **成本（论文）：** 结构件 PLA ~$30 或 SLS Nylon ~$50；含 7 舵机与电子件整手约 **$500**；PLA 版打印后约 **2 h** 可组装。
- **任务：** 开闭瓶盖、移液、泵吸、双手柄工具、拧螺丝、扣扳机、抓取内重定向等 **35** 物体；关节反馈做接触检测（无视觉闭环要求）。
- **迁移：** Franka Emika Panda 固定基座 → 人形；双臂各装一只 Cartesian Hand 做 **双手实验室操作**（如离心管开盖 + 移液）。

## 对 wiki 的映射

- [paper-cartesian-hand-linear-fingers](../../wiki/entities/paper-cartesian-hand-linear-fingers.md)
- [cartesian-hand-linear-fingers_arxiv_2609_25696.md](../papers/cartesian-hand-linear-fingers_arxiv_2609_25696.md)
- [cartesian_hand.md](../repos/cartesian_hand.md)

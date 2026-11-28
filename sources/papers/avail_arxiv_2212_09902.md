# 基于图像的灵巧操作：用子步骤引导真机自主强化学习 / Dexterous Manipulation from Images: Autonomous Real-World RL via Substep Guidance

- **类型：** paper
- **作者：** Kelvin Xu、Zheyuan Hu（共同一作）、Ria Doshi、Aaron Rovinsky、Vikash Kumar、Abhishek Gupta、Sergey Levine
- **机构：** UC Berkeley、Meta AI Research、University of Washington
- **论文：** [arXiv:2212.09902v1](https://arxiv.org/abs/2212.09902v1)（2022-12-19） · [PDF](https://arxiv.org/pdf/2212.09902v1)
- **出版：** ICRA 2023 · [IEEE DOI](https://doi.org/10.1109/ICRA48891.2023.10161493)
- **项目页：** [AVAIL 官方项目](https://sites.google.com/view/dexterous-avail/) · [补充资料归档](../sites/avail-dexterous.md)
- **核对范围：** arXiv v1 全文 12 页，正文 §I–VIII、图 1–8、表 I、附录 §IX–XII（实物与评测标准、仿真任务、算法 1–2、超参数与附加对照）；另核对官方项目页展示与开放入口，未运行训练或真机。
- **通读：** 已完成
- **入库日期：** 2026-11-28

## 可追溯证据

1. **§III–V、图 3–4：** 两路 84×84 RGB 与本体状态驱动 Sawyer 臂和 16 自由度四指手。人工摆出成功里程碑并标注下一步；用成功样例与在线经验学习奖励，另训练任务选择器以支持掉落后重抓。
2. **算法 2、§XI：** 启动时训练视觉任务图；外层按当前观测选技能，内层在固定时段收集经验，按技能分别维护策略、价值网络和回放。§V-B 用抽样描述任务选择，而 §V-C / 算法 2 写最可能任务（argmax）；按后者理解具体伪代码，不把论文写法冒充源码细节。
3. **图 1、5–8、15：** 刷子任务约 18 小时后各里程碑达到约 80% 或以上；图 5 报告约 36 小时无人值守训练后，挂钩约 95%、插接约 80%。学习任务图在该插接实验约 15 万步达稳定表现，手工顺序图约 20 万步。均非多平台或未见物体的端到端成功率。
4. **§VII、§IX：** 被动系绳阻止物体掉出工作区；没有完整动作示范不代表没有人工监督或装置。

- **对 wiki 的映射：** [AVAIL](../../wiki/entities/paper-avail-dexterous-manipulation-from-images.md)

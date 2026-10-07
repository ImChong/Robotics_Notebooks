# FineART / FineART-VLA 来源归档

- **论文：** [FineART: Fine-Grained Annotated Robotic Trajectory Dataset and Vision-Language-Action Model for Bimanual Manipulation](https://arxiv.org/abs/2609.36416)
- **PDF：** https://arxiv.org/pdf/2609.36416
- **代码实现：** [huggingface/lerobot](https://github.com/huggingface/lerobot)
- **官方实现说明：** https://github.com/huggingface/lerobot/blob/main/docs/source/fineart_vla.mdx
- **基础 checkpoint：** https://huggingface.co/lerobot/fineart_vla_base
- **作者：** Jade Choghari, Pepijn Kooijmans, Mansi Agarwal, Yusuf Umut Ciftci, Aseem Doriwala, Catherine Weaver, Mouli Sivapurapu, Kai Yang, Jackson Lee, Thomas Wolf, Pragna Mannam

## 摘录

FineART 数据包含 40,543 episodes、1,718 小时、533,913 个子任务标注，覆盖 151 项双臂操作任务。FineART-VLA 预测下一子任务，并以子任务条件动作生成进行中期训练。论文摘要报告空间消歧成功率从 32.0% 增至 100.0%；逐步人工子任务指导下未见长程任务从 16.0% 增至 76.0%；新机器人微调用量约为未中期训练基线的十分之一，并报告新硬件未见任务零样本泛化。

## 实现核对

LeRobot 官方文档将 FineART-VLA 描述为建立在 π0.5/Pi05 上，添加可训练语言头和运行时中间子任务生成。默认配方约 30% task→subtask、70% subtask-conditioned action。训练需子任务时间线标注；当前基础 checkpoint 为 lerobot/fineart_vla_base。

## 读数限制

- FineART 是数据资源；FineART-VLA 是策略模型。
- 76% 结果包含逐步人工子任务指导，不等同于模型全自主成功率。
- 数据效率与零样本结果依赖论文给定的新机器人协议。
- 论文与 LeRobot 当前实现文档共同作为资料来源；实现参数可能随主仓更新。

## [2026-11-08] ingest | EmbodiedRSI 假设引导的机器人持续学习

- **意图：** 收录哥伦比亚大学与普林斯顿大学 EmbodiedRSI 一手研究；澄清“机器人自己教自己”的模型/执行框架边界。
- **论文：** arXiv:2610.10498v1（公开 12 页全文已读）；整理 VOI 选择、Hypothesis Graph、Code-Skill 联合试验、PPO 记忆 Manager、仿真与 SO-101 真机结果。
- **可复现性：** 官方项目仓库只开放网站源码，未发现 EmbodiedRSI 可运行训练/部署实现；论文引用但未附的附录 A–I 另列公开材料缺口。
- **知识节点：** [EmbodiedRSI](../wiki/entities/paper-embodiedrsi.md)；从 [RoboDoctor](../wiki/entities/paper-robodoctor.md) 加入站内关联。
- **关键事实：** 冻结基础 VLA；真机零样本对象是仿真演化后的 harness；眼镜精密抓取任务出现回退。

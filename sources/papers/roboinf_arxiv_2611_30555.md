# RoboInF：机器人操作指令保真度基准 / A Benchmark for Instruction Fidelity in Robotic Manipulation

- **类型：** paper
- **作者：** Jiawei Guo、Yu Huang、Dijiang Dong、Yue Liu、Peiran Ren、Wei Shen
- **机构：** 上海交通大学、同济大学
- **论文：** [arXiv:2611.30555v2](https://arxiv.org/abs/2611.30555v2)（2026-11-24） · [PDF](https://arxiv.org/pdf/2611.30555v2)
- **项目页：** [官方演示与结果](https://sjtu-deepvisionlab.github.io/RoboInF/)
- **开放资源：** [官方代码归档](../repos/roboinf.md) · [RoboInF-4K 数据归档](../datasets/roboinf-4k.md)
- **核对范围：** v2 正文 §1–5、表 1–4，附录 A–G 的标注方法、任务条件与评测协议、策略训练、在线引导、推理代理和附加结果；对照项目页、代码及数据卡，运行公开评测输入校验，未跑策略 rollout。
- **通读：** 已完成
- **入库日期：** 2026-11-28

## 证据摘要

1. §3、图 2–3：在同一仿真初始状态只改指令，每任务 6–12 分支。每个保留场景要有所有分支均满足约束的专家轨迹；同场景分支必须进同一数据分割。
2. 表 1、附录 C：1,006 个候选场景中 566 个通过全部分支，共 4,242 轨迹。训练 2,520；验证和测试各 861。清晰和外观随机测试复用同一 115 场景和 861 指令，不能算成不同的物理条件。
3. 表 3–4：清晰、后训练、仅任务指令条件下，π0.5 为 52.4% JR / 79.3% TR，OpenWAM 为 34.0% / 70.0%。是八任务等权均值；不能把 861 轨迹简单池化，也不能将推理代理结果与学习策略视作同配置对比。

- **对 wiki 的映射：** [RoboInF](../../wiki/entities/paper-roboinf.md)

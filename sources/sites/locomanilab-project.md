# LocoManiLab 项目页

- **项目页：** https://locomanilab.github.io/
- **论文：** https://arxiv.org/abs/2610.30876
- **作者：** Dewei Wang、Jinrui Han、Jiyuan Shi、Chenyun Zhang、Xianbo Lu、Jiabao Wu、Jiqin Feng、Jipeng Kong、Ping Luo、Chenjia Bai、Xuelong Li
- **机构：** 中国电信人工智能（TeleAI）、中国科学技术大学、香港大学、浙江大学、上海交通大学、上海科技大学、Gamma Robotics
- **发布日：** 2026-10-23
- **代码 / 数据：** 2026-10-26 项目页均标注 “Coming soon”；2026-11-06 复核时官方仓库与 HF 数据集已公开（见下方开放状态）
- **公开评测：** Loco-Mani Arena 提供交互式策略与控制器结果；页面可下载部分汇总和逐任务 CSV
- **核查日期：** 2026-10-26（开放状态于 2026-11-06 复核）
- **一句话说明：** 面向人形移动操作的全栈测试床，将任务、示范数据、策略、全身控制、基准评测与仿真到真机比较纳入同一工作流。

## 项目页补充要点

- 基准包含 234 个任务定义（102 原子、132 组合）、13 个场景实例和 5 类家庭环境。
- 页面列出 14,160 段遥操作 episode、50 小时以上示范数据、12 种策略与 5 种全身控制器。
- 机器人接口以 Unitree G1 为共享具身，并展示 Dex3 灵巧手；策略执行阶段标注 50 Hz。
- 官网提供三个仿真与现实设置对应的任务演示，并提供 Kimodo 动作生成微调对比。
- **开放状态：** 2026-10-26 时 Code 与 Dataset 均为 “Coming soon”。2026-11-06 复核：官方仓库 <https://github.com/Gamma-Robot/LocoManiLab>（自有代码 MIT）与 HF 数据集 <https://huggingface.co/datasets/GammaRobotics/LocoManiLab>（公开、未设门槛）已上线，细节见[论文归档](../papers/locomanilab_arxiv_2610_30876.md)。

## 对 Wiki 的映射

- [LocoManiLab 实体页](../../wiki/entities/paper-locomanilab.md)
- [论文摘录](../papers/locomanilab_arxiv_2610_30876.md)
- [移动操作任务页](../../wiki/tasks/loco-manipulation.md)

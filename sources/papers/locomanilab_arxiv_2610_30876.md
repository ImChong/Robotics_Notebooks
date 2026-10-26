# LocoManiLab: A Unified Testbed for Data, Policies, Controllers, and Deployment in Humanoid Loco-Manipulation

> 来源归档（paper）；依据 arXiv v2 摘要页与论文项目页核对。

- **作者：** Dewei Wang、Jinrui Han、Jiyuan Shi、Chenyun Zhang、Xianbo Lu、Jiabao Wu、Jiqin Feng、Jipeng Kong、Ping Luo、Chenjia Bai、Xuelong Li
- **机构：** 中国电信人工智能（TeleAI）、中国科学技术大学、香港大学、浙江大学、上海交通大学、上海科技大学、Gamma Robotics
- **版本/日期：** arXiv:2610.30876v2，cs.RO；2026-10-23 修订
- **论文：** <https://arxiv.org/abs/2610.30876>
- **PDF：** <https://arxiv.org/pdf/2610.30876>
- **项目页：** <https://locomanilab.github.io/>
- **入库日期：** 2026-10-26

## 摘要式摘录

LocoManiLab 将人形移动操作的数据采集与生成、策略训练、评测、全身控制和实机部署放进同一测试床。任务集覆盖 5 类家庭场景中的 234 个原子与组合任务，论文报告 51.5 小时模拟遥操作数据。作者比较共享控制器下的 12 种策略、5 种控制器下的策略表现，并分析示范规模、数据来源混合、未见任务/场景和视觉扰动对泛化与鲁棒性的影响。

论文还微调一个人形动作生成模型以生成移动操作参考动作，研究合成动作与任务执行要求的对齐情况，以及它在数据生成链路中的作用。三个仿真任务与现实机器人设置相对应，用于跨仿真和真机比较策略表现。

## 项目页可核验信息

- **任务与场景：** 102 个原子任务、132 个组合任务、13 个场景实例，涵盖厨房、客厅、卧室、浴室和办公室；另有两个 Real2Sim 重建场景。
- **遥操作数据：** 项目页列出 14,160 段 episode、50 小时以上数据；策略执行频率展示为 50 Hz。机器人接口展示 Unitree G1 与 Dex3 灵巧手。
- **系统比较：** 12 种策略与 5 种全身控制器由 Loco-Mani Arena 汇总比较。论文摘要指出，未见任务与场景成功率下降，视觉扰动通常也会降低成功率，且不同策略的泛化和鲁棒性存在差异。
- **动作生成：** 项目页给出 Kimodo 在 30 个 task-scene 配对、每个模型 10 次试验上的总体任务成功率：预训练 39.7%、LoRA 微调 71.3%、全量微调 90.7%。这是项目页所列特定对比设置，不应外推为全部生成动作的总体成功率。
- **仿真到真机：** 官网提供开冰箱、货架到桌面搬运、传送带抓放三个任务的仿真与真机视频对照；论文摘要称三项任务用于检验仿真评测与实机表现的一致性。

## 开放状态

截至 2026-10-26，项目页的 **Code** 与 **Dataset** 均标注 “Coming soon”，未发现可公开克隆的官方代码仓或可下载的完整遥操作数据集。Loco-Mani Arena 可交互查看基准结果，页面也提供若干结果 CSV；这不等同于训练代码或完整示范数据已开放。

## 对 wiki 的映射

- [LocoManiLab 项目实体](../../wiki/entities/paper-locomanilab.md)
- [移动操作任务页](../../wiki/tasks/loco-manipulation.md)

# Fine-Tuning VLAs with Self-Demonstrated Generative Control for Multi-Task Manipulation（arXiv:2608.19490）

> 来源归档（ingest）

- **标题：** Fine-Tuning VLAs with Self-Demonstrated Generative Control for Multi-Task Manipulation
- **类型：** paper / vla / continual-learning / self-supervision
- **arXiv abs：** <https://arxiv.org/abs/2608.19490>
- **PDF：** <https://arxiv.org/pdf/2608.19490>
- **项目页：** <https://self-supervised-control.pages.dev/>（归档见 [`sources/sites/self-supervised-control-pages-dev.md`](../sites/self-supervised-control-pages-dev.md)）
- **机构：** UIUC（University of Illinois Urbana-Champaign）
- **作者：** Prachi Garg、Steve Xing、Prahit Yaugand、Saurabh Gupta、Derek Hoiem
- **发表 / 上传：** 2026-08-21（arXiv v1）
- **入库日期：** 2026-08-24
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md)（<https://mp.weixin.qq.com/s/e0yXB8Rz4ma3CCPX8HN2CQ>）

## 开源状态（步骤 2.5，2026-08-24）

- 打开 <https://self-supervised-control.pages.dev/>：视频结果与 RoboTwin 任务表齐全，**无 GitHub / Hugging Face / 权重链接**。
- **结论：** **确认未开源**（截至入库日仅项目页与 arXiv）。

## 摘录 1：问题

- 零样本 VLA（如 π₀.₅）在新机器人上因本体差异掉点；仅用新本体专家数据微调会遗忘指令跟随与行为先验。

## 摘录 2：方法

- 冻结零样本 VLA 在目标机器人上在线 rollout，将自生成轨迹与专家示范 **联合微调**（generative replay / self-distillation）。
- 自监督数据覆盖 pick-and-place 等预训练任务族，无需访问原始预训练数据。

## 摘录 3：评测要点

- **真机 ALOHA：** 5 任务族、59 prompts、120 场景；place 行为从 0%→55%（无专家 place 示范）；push 保留技能 60% vs oracle 5%；齿轮插入 30%→90%。
- **仿真 RoboTwin：** 旧任务 16.6%→70.6%；新任务 93%→98%。

**对 wiki 的映射：** [`wiki/entities/paper-self-supervised-control.md`](../../wiki/entities/paper-self-supervised-control.md)；交叉 [VLA](../../wiki/methods/vla.md)、[模仿学习](../../wiki/methods/imitation-learning.md)。

## 当前提炼状态

- [x] 项目页开源核查（确认未开源）
- [x] 升格 `wiki/entities/paper-self-supervised-control.md`

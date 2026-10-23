# SP-HLM：稀疏姿态驱动的人形全身控制（arXiv:2610.30044）

> 来源归档（ingest）

- **标题：** SP-HLM: Sparse-Pose-Driven Motion Generation and Whole-Body Control for Humanoid Loco-Manipulation
- **简称：** SP-HLM
- **类型：** paper / humanoid / loco-manipulation / sparse-pose / whole-body-control / VLA
- **arXiv：** <https://arxiv.org/abs/2610.30044>（v1，2026-10-21）
- **项目页：** <https://sp-hlm.github.io/>
- **作者：** Zhaohui Yang、Quanbao Lin、Chengqi Shi、Zhenglang Weng、Jicheng Lin、Boyu Chen、Tengjie Zhu、Guanzhu Ren、Jiahao Xu、Yang Feng、Xiaokang Yang、Yao Mu
- **入库日期：** 2026-10-23
- **一句话说明：** 用五个关键身体姿态表达任务空间意图，再以学习到的运动先验补全协调动作，连接高层 VLA 与人形全身控制。

## 摘要级要点

- **稀疏姿态接口：** 显式目标包含骨盆、左右手腕和左右脚踝五个身体部位；学习到的 latent motion prior 编码其余全身协调。
- **SP-VLA：** 根据多模态观察和语言指令预测稀疏姿态目标及手指关节位置，并采用异步 action-chunk 推理。
- **SP-MGM：** 通过两阶段蒸馏，把稀疏姿态目标实时重建为协调的全身动作。
- **SP-GMT：** 联合使用显式稀疏目标与隐式运动 latent 执行全身运动跟踪。
- **实验：** 摘要报告仿真及真实人形机器人上的跟踪、协调与 loco-manipulation 结果优于对照方法；具体数字应以论文实验表为准。
- **项目页/代码核查：** arXiv 提供项目页链接；本次项目页抓取未成功，故代码开放状态待进一步确认，不据此断言未开源。

## 核心论文摘录（MVP）

### 1) 显式任务空间目标 + 隐式全身先验

- **链接：** [arXiv 摘要](https://arxiv.org/abs/2610.30044)
- **摘录要点：** 五个身体关键点给出任务相关空间约束，预训练运动编码器的 latent 提供稀疏约束未描述的全身协调信息。

### 2) 三模块分层控制

- **链接：** [arXiv 摘要](https://arxiv.org/abs/2610.30044)
- **摘录要点：** SP-VLA 产出稀疏目标，SP-MGM 生成协调动作，SP-GMT 跟踪目标及 motion prior；高层不必直接输出完整全身关节配置。

### 3) 官方项目页核查

- **链接：** [SP-HLM 项目页](https://sp-hlm.github.io/)
- **摘录要点：** arXiv 将该地址列为补充结果与视频的项目页。本次抓取不可用，代码链接与数据发布状态留待可访问时核查。
- **对 wiki 的映射：** 待实体页升格后链接本归档及 [SP-HLM 项目页归档](../sites/sp-hlm.md)。

## 对 wiki 的映射

- 计划沉淀实体页：`wiki/entities/paper-sp-hlm.md`（待建）
- 项目页归档：[`sources/sites/sp-hlm.md`](../sites/sp-hlm.md)
- 相关主题：稀疏动作表示、VLA、运动生成、全身控制、loco-manipulation

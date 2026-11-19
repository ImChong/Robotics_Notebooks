# Heave-Ho — 官方项目页归档

- **页面：** https://heaveho-humanoid.github.io/
- **论文：** https://arxiv.org/abs/2611.22275
- **代码仓库：** https://github.com/heaveho-humanoid/Heave-Ho
- **机构：** 南加州大学（University of Southern California）；VinMotion
- **入库：** 2026-11-19
- **开源状态：** 仓库公开但实际训练/推理/评测代码和权重尚未发布；README 明确计划于 2026 年 12 月底放出代码，不能按已开源项目标注。
- **核对范围：** 核对项目主页全文（Abstract、How it works、语音机器人/人类领队演示、双 G1 及多人/混合团队视频说明、HH-Bench 任务与结果、Table 中真机成功率和高低层结果）；对照 arXiv HTML v2 及官方 GitHub README 核对发表和开源状态。
- **通读：** 已完成

## 项目页直接证据

1. 页面显式写明训练团队为两台相同人形、部署最多十台、HH-Bench 12 项任务、机器人之间 0 网络消息。
2. 领导者视觉语言规划模型决定技能，再通过扬声器广播语音；跟随者使用机载麦克风识别，在统一技能调度格启动。页面展示英文与中文语音指令，以及人类或机器人领导的案例。
3. 实物演示涵盖两台 G1 搬箱穿窄门、L 形走廊、卡车交接箱、精确放置；扩展到四台 G1 搬桌、人机合作、PM01 与 G1 混合。
4. 页面公布低层 HH-Bench 平均 94.41%，五项高层任务规划器对比、真机十组共计 92/100；原文逐项统计以论文表格为准。
5. 项目页“Code”指向 GitHub 仓库，但仓库当前仅有 README 与 assets，代码发布时间以仓库说明为准。

## 映射

- [论文完整通读归档](../papers/heave_ho_arxiv_2611_22275.md)
- [官方仓库状况](../repos/heave-ho.md)
- [唯一实体页](../../wiki/entities/paper-heave-ho.md)

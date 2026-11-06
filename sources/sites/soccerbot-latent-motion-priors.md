# SoccerBot 项目页

> 来源归档

- **标题：** SoccerBot: Playing Soccer Like Humans Do
- **类型：** site
- **链接：** <https://soccerbot-humanoid.github.io/>（2026-11-06 HTTP 200）
- **论文：** <https://arxiv.org/abs/2611.06564>
- **机构 / 会议：** USTC、BIGAI、上海交通大学、北京大学；CoRL 2026
- **入库日期：** 2026-11-06

## 页面内容

- 摘要、方法三阶段简介（与论文一致，写明"约 2.5 小时"可行参考运动）、真机四技能视频（Move/Dribble/Trap/Kick，含 "15 m/s command" 的踢球演示）、长时序任务（Dribble→Kick，Trap→Kick→Trap）、仿真多速度视频、四机器人"写 CORL"接力视频、BibTeX。
- **Interactive demo / SoccerBot Playground**（`/play/`，HTTP 200）：浏览器内 MuJoCo WASM + ONNX Runtime Web，可带球与点球射门。
- 页面布局注明借鉴 Nerfies。

## 开源状态（步骤 2.5，2026-11-06）

- 页面**没有**指向训练代码仓库、数据集或权重下载的链接，未见 "coming soon" 字样。
- `git ls-remote` 对 `soccerbot-humanoid/soccerbot`、`soccerbot-humanoid/SoccerBot` 均不可达（要求认证，即不存在或私有）。
- 论文视频页 Playground 内含一个 `policy.onnx`（见 [repos 归档](../repos/soccerbot-latent-motion-priors.md)），但不含训练管线；许可未声明。
- **结论：** 训练 / 蒸馏 / 部署代码与动捕数据：未发布；权重：仅 playground 的单个 ONNX 策略，无许可声明。

## 对 wiki 的映射

- [paper-soccerbot-latent-motion-priors](../../wiki/entities/paper-soccerbot-latent-motion-priors.md)

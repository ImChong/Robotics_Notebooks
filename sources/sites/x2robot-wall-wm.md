# WALL-WM Project Page（自变量机器人）

> 来源归档

- **标题：** WALL-WM — Carving World Action Modeling at the Event Joints
- **类型：** site / project page（技术报告页）
- **URL：** <https://x2robot.com/pages/wm>（英文：<https://x2robot.com/en/pages/wm>）
- **论文：** [arXiv:2606.01955](https://arxiv.org/abs/2606.01955)（v1 2026-06-01；v2 2026-09-06，[HTML](https://arxiv.org/html/2606.01955)）
- **技术报告 PDF（官网）：** <https://x2robot.com/api/files/file/WALL-WM.pdf>
- **代码：** <https://github.com/X-Square-Robot/wall-wm>（Apache-2.0）
- **机构：** 自变量机器人（X Square Robot）；作者署名 *X Square Robot Team*
- **官方页日期：** 2026-05-29（页眉 *X Square Robot · Tech Report · May 29, 2026*）
- **入库日期：** 2026-10-09
- **一句话说明：** 官方技术报告页给出「事件而非定长 chunk」的核心论点、三大支柱（事件级对齐 / 双推理模式 / 规模化基建）、WorldArena 视频生成表与真机四套件 Task Progress 汇总。页面是 Next.js 前端渲染，正文藏在 RSC payload 内嵌的 HTML 里，静态抓取需解析 `self.__next_f` 才能拿到文字。

## 开源核查（2026-10-09）

| 入口 | 状态 |
|------|------|
| Homepage | 已上线：<https://x2robot.com/pages/wm> |
| Paper | arXiv:2606.01955 v2（2026-09-06），许可 CC BY-NC-ND 4.0；官网另挂 PDF（2026-10-09 GET 返回 200/206） |
| Code | **已开源**：[X-Square-Robot/wall-wm](https://github.com/X-Square-Robot/wall-wm)，单个提交 `wall-wm opensource`（2026-07-02）。仓里有 Wan 视频 DiT + 动作塔建模、FSDP 训练器（含 DMuon 优化器）、WebSocket 推理服务、RoboTwin 评测、事件模式开环评测、LIBERO LeRobot 微调配置、事件后训练配置 |
| Weights | **未发布**：README 写 *Pretrained WALL-WM checkpoints: coming soon* at `huggingface.co/x-square-robot`；2026-10-09 查 HF API，该组织下只有 wall-oss / X-Tokenizer / X2 语音等模型，没有 WALL-WM 权重 |
| Data | **未发布**：论文的预训练语料（自采遥操作 + XRZero-G0 无本体 + 公开数据混合）没有放出；仓库只带 1 条示例 episode `put_spoon_to_bowl`（三视角 mp4 + 轨迹 + 事件字幕）。HF 上的 `x-square-robot/XRZero-G0-3K` 来自同一采集装置，但官方没说它是 WALL-WM 的训练集 |
| 页面 GitHub 按钮 | 官方页的 *Code on GitHub* 和 BibTeX `url` 指向 [wall-x](https://github.com/X-Square-Robot/wall-x)（WALL-OSS 仓），**不是** wall-wm；论文 HTML 的 `[Code]` 与 wall-wm README 则互指 wall-wm |

> 推测：Staircase 隐式 CoT 解码器、DMD 少步蒸馏、部署用 FP8 PTQ 这三块，按关键字在仓库 Python 代码里都没搜到（只找到 `Qwen35TextEncoderAdapter` 文本编码适配器和 FP8 训练开关），大概率还没随仓发布。

## 页面内容要点

- **核心论点**：固定 chunk 按时钟切，语义事件按具身动力学切；语言给事件命名，视频承载事件的时序演化，动作负责执行。
- **三大支柱**：① *Event-level alignment*，原子训练单元是 action-centered semantic event；② *Two modes, one backbone*，事件模式支持任意长度执行段，统一模式用 VLM + Staircase Decoding 做定长 chunk；③ *Built to scale*，DMuon、融合 kernel、多事件序列打包、聚类平衡采样，再加 FP8 + DMD 蒸馏把推理压到实时控制延迟。
- **页面汇总数字**：WorldArena 协议视频生成 Semantic Align **0.886**、Interaction **0.434**、Trajectory Acc. **0.234**；CO3Dv2 Point Err **0.271** / Depth Err **0.132**；真机四套件平均 Task Progress **58.30**（页面标作 *Real-Robot Core15 · Basic*），对照 π0.5 37.76、DreamZero 31.54。
- **示例生成**：浇花（调喷壶喷嘴）和泡茶（调整杯柄位置），与 Wan2.1 / Wan2.2 同帧同指令对比。

## 对 wiki 的映射

- 沉淀到 **[`wiki/entities/paper-rcl-2606-01955-wall-wm-carving-world-action-modeling-at-the-eve.md`](../../wiki/entities/paper-rcl-2606-01955-wall-wm-carving-world-action-modeling-at-the-eve.md)**
- 代码归档：[`sources/repos/wall-wm.md`](../repos/wall-wm.md)
- 清单摘录：[`sources/papers/rcl_awesome_wam_2606_01955_wall-wm-carving-world-action-modeling-at.md`](../papers/rcl_awesome_wam_2606_01955_wall-wm-carving-world-action-modeling-at.md)

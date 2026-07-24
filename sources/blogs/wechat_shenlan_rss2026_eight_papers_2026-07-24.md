# 顶会 RSS 2026 释放明确信号：8篇力作聚焦底层控制、轨迹优化、物理硬件

> 来源归档（blog / 微信公众号）

- **标题：** 顶会 RSS 2026 释放明确信号：8篇力作聚焦底层控制、轨迹优化、物理硬件
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/qjhBjBqTYHcfnndPFNVb-g
- **发表日期：** 2026-07-24
- **入库日期：** 2026-07-24
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_rss2026_eight_papers_2026-07-24/`](../raw/wechat_shenlan_rss2026_eight_papers_2026-07-24/)
- **一句话说明：** 从 **RSS 2026 Final List 8 篇**（含 Outstanding Paper / Student / Systems）归纳「回归物理」信号：底层控制、轨迹优化、执行器/硬件与仿真耦合；本库 **复用已有完整节点，不重复造页**。
- **姊妹报道：** [量子位 RSS 2026 颁奖盘点（2026-07-16）](./wechat_qbitai_rss2026_awards_2026-07-16.md) — 同一 Final List，奖项与 ToT 覆盖更全。

## 核心摘录（归纳，非全文）

### 会议背景（文内数字）

| 指标 | 文内表述 |
|------|----------|
| 录用 | 210 篇 |
| 接收率 | 不足 30%（较上年约 +2%） |
| 叙事主线 | 集体「回归物理」：控制 / 轨迹优化 / 物理硬件 / 仿真引擎 |

### Final List 8 篇 → 本库节点（全部已有 · 非 stub）

| 角色 | 论文 | arXiv | wiki（复用，不新建） |
|------|------|-------|----------------------|
| Outstanding Paper | FlashSAC | [2604.04539](https://arxiv.org/abs/2604.04539) | [flashsac（方法页）](../../wiki/methods/flashsac.md) |
| Outstanding Student Paper | Muninn | [2605.09999](https://arxiv.org/abs/2605.09999) | [paper-muninn-…](../../wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md) |
| Outstanding Systems Paper | NeuralActuator | [2607.11734](https://arxiv.org/abs/2607.11734) | [paper-neuralactuator-…](../../wiki/entities/paper-neuralactuator-neural-actuation-modeling.md) |
| Finalist | OAT | [2602.04215](https://arxiv.org/abs/2602.04215) | [paper-oat-…](../../wiki/entities/paper-oat-ordered-action-tokenization.md) |
| Finalist | DAPL（外在灵巧） | [2603.09882](https://arxiv.org/abs/2603.09882) | [paper-dapl-…](../../wiki/entities/paper-dapl-extrinsic-dexterity-clutter.md) |
| Finalist | cuNRTO | [2603.02642](https://arxiv.org/abs/2603.02642) | [paper-cunrto-…](../../wiki/entities/paper-cunrto-gpu-robust-trajectory-optimization.md) |
| Finalist | Unified Fluid–Robot Multiphysics | [2506.05012](https://arxiv.org/abs/2506.05012) | [paper-unified-fluid-…](../../wiki/entities/paper-unified-fluid-robot-multiphysics-swimming.md) |
| Finalist | Automated Facial Mechanisms | [2607.11688](https://arxiv.org/abs/2607.11688) | [paper-automated-facial-…](../../wiki/entities/paper-automated-facial-mechanisms-animatronic.md) |

### 文内要点速记（编译）

1. **FlashSAC** — 少更新 + 大模型/高吞吐 + 范数约束；G1 平地盲走约 **20 min** vs PPO **~3 h**。
2. **Muninn** — 免训练缓存封装器：探针监测内部表征，预算耗尽才调完整 denoiser。
3. **NeuralActuator** — Transformer 执行器模型：力矩 / 外力 / 工况；BC 抓取放置 **80%→92.5%**。
4. **OAT** — 有序动作分词，粗到细，可按算力预算截断生成深度。
5. **DAPL** — 显式世界模型动力学 + RL；杂乱场景外在灵巧（推拨翻转）。
6. **cuNRTO** — CPU 外层线性化 + GPU 内层鲁棒优化；最高约 **139.6×** 加速。
7. **流固耦合** — 统一最小作用量多物理；游动/抓取仿真墙钟提升约两个数量级。
8. **面部机构自动合成** — 单张 2D 人脸 → 可制造无干涉连杆头，十余分钟级设计。

## 对 wiki 的映射

- **不新建** `paper-*` / 方法页：8 篇均已在量子位姊妹 ingest 或更早入库为 `status: complete`。
- FlashSAC 保持唯一枢纽为 [`wiki/methods/flashsac.md`](../../wiki/methods/flashsac.md)（**不**另造 `paper-flashsac`，避免重复节点）。
- 本次仅：归档本博客 → 各页 `sources` / `参考来源` 回链 → `log.md`。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] Final List 8 篇独立完整节点核查（0 stub / 0 新建）
- [x] 与量子位 RSS 2026 报道交叉引用

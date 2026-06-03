# yutoshibata07.github.io/AssistMimic（AssistMimic 项目页）

> 来源归档（ingest）

- **标题：** AssistMimic — CMU × Keio
- **类型：** site / project-page
- **官方入口：** <https://yutoshibata07.github.io/AssistMimic/>
- **入库日期：** 2026-06-03
- **一句话说明：** 论文配套站点：强调 **MARL + 单人 motion prior + recipient-adaptive retargeting + contact-promoting reward** 首次在 **高接触 assistive MoCap** 上实现 physics tracking；含 **kinematic baseline 失败** 对比视频、Inter-X / HHI-Assist 定量表、扩散生成交互跟踪与 **unseen interaction** 泛化演示。

## 页面公开信息（检索自 2026-06-03）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://yutoshibata07.github.io/AssistMimic/> |
| 论文 abs | <https://arxiv.org/abs/2603.11346> |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **核心叙事**：双人 assistive 交互 = **multi-agent RL**；需 **joint training** 而非 kinematic replay / frozen recipient。
2. **方法四块**：Multi-Agent RL formulation；Motion prior initialization（PHC）；Dynamic reference retargeting；Contact-promoting rewards。
3. **定性对比**：Kinematic replay 导致 recipient **独立站起** 或 **严重 interpenetration**；站点以视频强调 **MARL 必要性**。
4. **定量表（specialist）**：
   - Inter-X：AssistMimic SR **74.9%** vs Sequential **62.4%**；去 weight init **0%**。
   - HHI-Assist：AssistMimic SR **85.8%**；去 contact reward 在 unseen hip torque 下 SR **27.7%**（reward hacking 风险，站点标注 †）。
5. **扩展演示**：扩散模型生成的 dense interaction kinematics → physics tracking；**support-by-arm** 等 **训练未见交互类别** 的 zero-shot 跟踪。

## 对 wiki 的映射

- [`wiki/entities/paper-assistmimic.md`](../../wiki/entities/paper-assistmimic.md) — 方法栈、实验与局限归纳页（含 MARL 落地语境）。

## BibTeX（站点页提供）

站点提供 `@inproceedings{shibata2026assistmimic, ... CVPR 2026}`；正式引用以 arXiv / 录用版本为准。

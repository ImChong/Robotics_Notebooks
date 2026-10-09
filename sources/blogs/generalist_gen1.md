# GEN-1: Scaling Embodied Foundation Models to Mastery（Generalist AI）

> 来源归档（blog / Generalist AI 官方）

- **标题：** GEN-1: Scaling Embodied Foundation Models to Mastery
- **类型：** blog
- **作者 / 组织：** Generalist Team / Generalist AI
- **原始链接：** <https://generalistai.com/blog/gen-1>
- **发表日期：** 2026-04-02（页面署名 April 2, 2026；Citation 写 Apr 2026）
- **入库日期：** 2026-10-09
- **抓取方式：** `curl` 抓取官方页静态 HTML（服务端渲染，正文可直接抽取）；视频与图表数值仅取正文与图注文字
- **一句话说明：** Generalist **GEN-1** 发布博文：在 GEN-0 基础上继续扩大数据与算力并叠加算法改进，宣称首次跨过「简单物理任务 **mastery**」阈值——三项灵巧任务平均成功率 **99%**（GEN-0 微调 **64%**、无预训练 **19%**），任务完成速度约为此前 SOTA 的 **~3×**（折盒 12.1 s，2.8×），且每项结果只用约 **1 小时机器人数据**；预训练数据 **>50 万小时**、来自人类可穿戴设备、**不含机器人数据**。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-10-09） |
|----|-------------------------|
| 本篇博客 / 项目页 | **无**独立项目页或技术报告 / arXiv；入口为 <https://generalistai.com/blog/gen-1> |
| 代码 / 权重 | **确认未开源**：正文无 GitHub / Hugging Face 链接；访问方式为「Early access partners」+ `partnerships@generalistai.com` 申请；Hugging Face `author=generalistai` 模型列表为空（2026-10-09 查询） |
| 数据集 | **未公开**（in-house 物理交互数据，>50 万小时） |
| 可信度边界 | 产业官方博客，非 peer-reviewed；成功率、速度、数据量均为 **自报**，评测协议与试次数未公开 |

## 核心摘录（归纳，非全文）

### 定位与主张

- GEN-1 是「实时输出动作的大型多模态模型」；作者称其为 **首个跨过简单物理任务 mastery 阈值** 的通用物理 AI 模型，可在广泛任务上 **具备商业可行性**（作者立场）。
- 进步来源：GEN-0 基础上 **进一步扩大数据与算力** + **算法改进**；正文称是对具身基础模型的「full redesign」，**从零训练**（trained from scratch）。
- 作者把 GEN-0 → GEN-1 类比为 GPT-2 → GPT-3：前者证明可扩展的多任务路径，后者在部分任务上跨过经济可用门槛。
- GEN-0 发布约在 GEN-1 之前 **五个月**（正文「Five months ago」；对比基线标为 **November 2025 版 GEN-0**）。

### Mastery 定义（三要素）

| 要素 | 博客定义 |
|------|----------|
| **Reliability（可靠性）** | 跨任务 / 系统 / 环境的稳健可重复成功，而非偶尔成功一次 |
| **Speed（速度）** | 以 **任务完成时间** 而非电机速度计；高速时非准静态效应（速度项、摩擦变化、运动模糊）上升 |
| **Improvisation（即兴智能）** | 意外情境下创造性恢复；作者认为这是机器人领域过去最缺的一项，依赖 **physical commonsense** |
| 附加维度 | 评估 mastery 时须同时看 **达到该性能所需的任务数据量** |

### 可靠性（自报）

| 任务 | 连续无干预表现 | GEN-1 | GEN-0（2025-11 版微调） | GEN-0 架构无预训练 |
|------|----------------|-------|--------------------------|--------------------|
| 扫地机器人维护（Servicing Robot Vacuum） | 200+ 次连续 | 99% | 50% | 2% |
| 折盒（Folding Boxes） | 200 次连续 | 99% | 81% | 13% |
| 手机装盒（Packing Phones） | 100 次连续 | 99% | 62% | 42% |
| **平均** | — | **99%** | **64%** | **19%** |
| 汽车零件配套（Kitting Auto Parts） | 连续 1 小时以上无干预 | — | — | — |
| 叠 T 恤 | 连续 86 次 | — | — | — |
| 积木装箱（Packing Blocks） | 连续 1,800+ 次 | — | — | — |

- 平均值与三项任务数字算术一致（(50+81+62)/3≈64，(2+13+42)/3≈19）。
- 图注称：GTC 上展示过同类任务的更新版 GEN-0 预训练模型（原文写「March 2025 at GTC」，与「November 2025 版」时间线不符，**推测** 为 2026 年 3 月 GTC 的笔误）。

### 速度（自报，视频 1× 实时、全自主）

- **折盒：** GEN-1 约 **12.1 s**；GEN-0 与 π0 在相同纸盒上约 **34 s**，π\*0.6 在相近但不同纸盒上相近 → **2.8×**。计时仅从「为折叠而触碰盒子」到「折叠完成」。
- **手机装壳：** **15.5 s**，为 GEN-0 的 **2.8×**。
- 摘要口径「~3× faster than state of the art」即上述 2.8× 的取整。
- 可 **快于示范**（作者称），并能在高速下对新物体物理作出反应。
- 速度来源（作者归因）：① **从经验中学习（RL）**；② 推理方式演进 **Harmonic Reasoning**（未给机制细节）；③ 可穿戴采集设备带来大量 **高速完成任务** 的预训练数据，而遥操作因缺力反馈、延迟、视野受限产出较慢数据。

### 即兴智能（定性）

- 汽车零件配套：垫圈被碰歪后，模型可 **放下重抓**、**部分插入缝隙借外部灵巧（extrinsic dexterity）重抓**、或 **换另一只手做双手手内重抓**。
- 大型可变形物体进入非常规构型时自行恢复。
- 作者称这些行为「远在训练分布之外」（无定量指标）。

### 系统组成与数据

- 组成：**预训练改进**（改善预训练算力效率曲线）+ **后训练技术** + **从经验学习（RL）** + **多模态人类引导（multimodal human guidance）** + **新推理期技术**。
- 作者称 GEN-1「更准确地说是一个 **系统**」：推理与 harness 等系统级组件对性能至关重要，不只是一组权重。
- **数据效率：** 部分测试中以 **1/10 的任务数据与微调步数** 达到 GEN-0 相当性能；文中各结果只用 **约 1 小时机器人数据**。
- **预训练不含机器人数据**：基座来自 **人类佩戴低成本可穿戴设备** 完成数百万种活动的数据；故适配新任务时 **同时首次适配该机器人具身与该任务**。
- 数据规模：**>50 万小时**（half a million hours）高保真物理交互数据。
- 作者对比：此前成功率 >90% 的通用机器人模型依赖昂贵、难扩展的大规模遥操作数据；GEN-1 被称为「不需大规模遥操作或仿真数据即可达高 mastery」的存在性证明（作者立场）。

### 工程基础设施（Looking Ahead，定性）

- 重设计分布式训练基础设施，以 **PB 级** 物理交互数据为一等公民。
- 训练稳定性、**自定义 kernel**、为实时推理发明 **新形式 paged attention**、后训练（理论 RL + 多模态人类引导基础）、更平滑精准的控制。
- 设计新硬件并在新地区 **发运数千只机器人手**（采集端扩展）。

### 局限与对齐（作者自述）

- 并非所有尝试过的任务都达 99%+；部分真实场景需要更高成功率或速度。
- 预期下一代模型扩展可 mastery 的任务范围，且随基座提升 **每任务数据需求下降**（预测）。
- **对齐：** 涌现即兴（摇袋让物体落位、整理错放物、去接下落物体）是有真实后果的物理动作；成功定义是任务 / 流程 / 用户特定的——涌现行为可能是优势也可能是隐患，需改进对齐方法以 **精确引导** 到用户所需行为（引 Inference-Time Policy Steering, Wang et al. 2025）。

### 可用性

- 发布当日向 **Early access partners** 开放；其余需邮件联系 partnerships。

## 对 wiki 的映射

- [generalist-gen1](../../wiki/entities/generalist-gen1.md) — 本篇升格实体页（GEN-1 发布）
- [generalist-ai-robotics](../../wiki/entities/generalist-ai-robotics.md) — 公司入口页 GEN 时间线
- [generalist-gen1-thousand-hands](../../wiki/entities/generalist-gen1-thousand-hands.md) — 2026-07 GEN-1 多末端后续博文
- [generalist-gen15-one-shot](../../wiki/entities/generalist-gen15-one-shot.md) — 后继 GEN-1.5（与 GEN-1 并行启动预训练）
- [physical-commonsense-generalist](../../wiki/entities/physical-commonsense-generalist.md) — 即兴智能依赖的物理常识叙事
- [embodied-scaling-laws](../../wiki/concepts/embodied-scaling-laws.md) — GEN-0 → GEN-1 规模叙事

## 可信度与使用边界

- 官方博客 + 精选视频；无技术报告、无第三方基准、无试次数 / 置信区间。
- 「首个」「commercial viability」「SOTA」均为作者表述；速度对比跨团队、跨纸盒（π\*0.6 为「相近但不同」纸盒），且只计折叠段时间。
- 「预训练不含机器人数据」与「约 1 小时机器人数据后训练」是可检验但外部无法复核的主张。

## Citation

```bibtex
@article{generalist2026gen1,
  author = {Generalist Team},
  title = {GEN-1: Scaling Embodied Foundation Models to Mastery},
  journal = {Generalist AI Blog},
  year = {2026},
  note = {https://generalistai.com/blog/gen-1}
}
```

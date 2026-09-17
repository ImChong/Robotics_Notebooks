# Helix 2.5: Zero-Shot 30-Home Generalization

> 来源归档（blog / Figure AI 官方）

- **标题：** Helix 2.5: Zero-Shot 30-Home Generalization
- **类型：** blog
- **作者：** Figure AI
- **原始链接：** https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization
- **发表日期：** 2026-09-17
- **入库日期：** 2026-09-17
- **抓取方式：** 官方新闻页直接抓取（WebFetch）
- **一句话说明：** Figure 发布 **Helix 2.5**：在 **Index** 人类行为数据上预训练的单基座模型，经任务数据微调后产出整理客厅、叠毛巾、铺床三项 **全身 locomanipulation** 行为；在 **30 套从未采集数据的湾区家庭** 中 **零样本** 部署同一 checkpoint，盲评成功率 **56%**（对照随机初始化仅 **9%**），并报告人→人形 **迁移 scaling law**。

## 核心摘录（归纳，非全文）

### 问题与定位

- **Helix 02 局限：** 已展示洗碗机卸载、物流 200 h 自主等 **长时程全身** 能力，但数据来自 **机器人将工作的环境**。
- **Helix 2.5 问题：** 人形能否进入 **从未见过的家庭**，无需采集/微调/适配，立即 **全身自主** 工作？
- **与 Helix 02 初始化差异：** Helix 2.5 **完全在 Index 上从随机初始化预训练**；Helix 02 从预训练 VLM 起步。

### 管线概要

1. **Index 预训练** — Figure 全球规模人类行为数据集（单基座）。
2. **任务微调** — 同一基座适配三项 distinct 行为（locomotion + 刚/软体操作 + 双手 + 主动感知）。
3. **零样本家庭评测** — 30 套湾区 unseen homes + unseen objects；**固定单一 checkpoint**，无权重适配。

### 关键结果（官方自报）

| 指标 | 数值 / 结论 |
|------|-------------|
| 零样本家庭数 | **30** 套真实家庭，均未采集任何数据 |
| 行为数 | **3**（整理客厅、叠毛巾、铺床） |
| Index 预训练贡献 | 盲评成功率 **9%**（scratch）→ **56%**（Index-init），约 **6×** |
| 任务规格数据效率 | 相对 Helix 02 代表行为 **减半** 任务数据，泛化范围 **30×** 家庭 |
| Scaling law | Index 预训练数据 **8×** 嵌套子集；下游 action-prediction loss 可 **四位小数** 预报最大 run（误差 **0.54%** 全范围变异） |
| Index 吞吐（同期） | 约 **35 分钟/秒** 新人类经验；已承诺 **$35 亿** 算力训练 Helix |

### 三项评测任务（零样本对象/布局）

| 任务 | 成功标准（无部分分） |
|------|----------------------|
| **Living Room Tidy** | 场景中 **13–15** 个散落玩具全部捡起放入篮中 |
| **Towel Folding** | 所有毛巾折叠并放入篮中 |
| **Bed Making** | 两个枕头与被子角放到床头上 1/3；被子拉平 |

**「零样本」精确定义（文内）：** 指 **评测环境与被操作物体** unseen；任务通过 **在其他处采集的微调数据** 指定。评测家庭内 **零采集**；评测玩具/毛巾/床品 **未出现在任务规格数据**；使用各家庭现有沙发/床/折叠面。

### 消融：Index 预训练

- 两策略：**相同** 任务规格数据、架构、优化、超参、评测；唯一差别为 **是否 Index 初始化**。
- 成功 = **整任务完成**（非部分玩具/单条毛巾）。

### 定性：全身自纠错

- 陌生环境中后退重定位、换 stance、绕床修正折叠等 **long-horizon self-correction**；作者归因 Index 预训练。

### 局限（作者边界）

- **未宣称通用家庭机器人已解决**；仅为「全身智能可从人类经验学习并迁移」的 **首批证据**。
- 定量除上述外多为演示级；**无 peer review**。
- 任务分布受控（三项家务式 locomanipulation）；超时与安全人工介入计失败。

## 开源核查（入库日 2026-09-17）

- **项目页 / 新闻页 / [Helix 专题](https://www.figure.ai/helix)：** 无 GitHub、Hugging Face、权重或数据集公开链接。
- **结论：** **未开源** — 模型、Index 数据与训练栈均不可公开复现；以 Figure 官方博客为准。

## 对 wiki 的映射

- [helix-25](../../wiki/entities/helix-25.md)（Helix 2.5 系统实体 + 预训练→微调→零样本家庭评测流程图）
- 交叉：[Figure AI](../../wiki/entities/figure-ai.md)、[VLA](../../wiki/methods/vla.md)、[VLA 演进技术地图](../../wiki/overview/vla-evolution-lineage.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[Foundation Policy](../../wiki/concepts/foundation-policy.md)

## 可信度与使用边界

- **公司官方博客**，非 peer-reviewed；成功率与 scaling law 为 **Figure 自报**，待独立复现。
- **Index 与 Helix 2.5 权重未公开**；「35 分钟/秒」「$35 亿算力」为同期商业叙事。
- 30 家庭为湾区受控评测；**不宜外推** 为任意地理/户型/物体分布下的部署保证。

## 参考来源

- [Figure AI · Helix 2.5 官方新闻](https://www.figure.ai/news/helix-2-5-zero-shot-30-home-generalization)

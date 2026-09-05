---
type: entity
tags: ["paper", "vla", "physical-intelligence", "flow-matching", "hmi-papers"]
status: complete
updated: 2026-09-05
arxiv: "2504.16054"
code: https://github.com/Physical-Intelligence/openpi
venue: "HMI curated · 2025"
summary: "π0.5（HMI P059）：预训练用 FAST 离散动作吃异构数据，后训练再为目标本体接入连续 flow 专家；推理时先出语义子任务再高频生成动作块。"
related:
  - ../methods/π0-policy.md
  - ../methods/pi07-policy.md
  - ../methods/vla.md
  - ../concepts/foundation-policy.md
  - ../entities/humanoid-motion-intelligence.md
  - ./paper-emergent-transfer-cross-config.md
  - ./paper-galaxea-g05.md
  - ./paper-kai0.md
  - ./paper-spd.md
  - ./paper-indi.md
  - ./paper-flashvla.md
  - ./paper-clap-cross-embodiment.md
sources:
  - ../../sources/papers/hmi_p059_pi05-open-world-vla.md
  - ../../sources/repos/humanoid-motion-intelligence.md
  - ../../sources/papers/chi0_kai0_arxiv_2602_09021.md
  - ../../sources/papers/flashvla_arxiv_2608_27384.md
  - ../../sources/papers/clap_arxiv_2608_27406.md
---

# π0.5（HMI P059）

**π0.5**（*π0.5: A Vision-Language-Action Model with Open-World Generalization*，2025，[arXiv:2504.16054](https://arxiv.org/abs/2504.16054)）收录于具身智能研究室 [论文与项目总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md) **P059**，主分类为 **世界模型、VLA与Agent**。本页为本库独立详情节点（编译自策展解读与公开元数据，非原文镜像）。

## 一句话定义

预训练用 FAST 离散动作吃异构数据，后训练再为目标本体接入连续 flow 专家；推理时先出语义子任务再高频生成动作块。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型 |
| FAST | Frequency-space Action Sequence Tokenization | 离散动作 token 化 |
| FM | Flow Matching | 连续动作专家生成 |
| BC | Behavior Cloning | 示范数据监督主线 |

## 为什么重要

- Flow matching很适合精细连续控制，但不方便与文本、网页问答和只有语义标注的数据用同一个自回归目标混合。π0.5先用FAST将连续动作压缩成离散token，与多本体轨迹、高层语义预测和web数据共同预训练。这个阶段的目标不是达到最精细的底层控制，而是建立场景、任务、本体和动作语义之间的广泛对应。
- 在 HMI 六条技术路线中属于 **世界模型、VLA与Agent**，补齐「总索引有条目、本库无下钻页」的缺口。
- 与相邻方法对照时，优先看问题设定与接口，而不是只记算法名。

## 核心信息

| 字段 | 内容 |
|------|------|
| HMI ID | P059 |
| 年份 | 2025 |
| 分组 | 世界模型、VLA与Agent |
| 开源状态 | 部分开源（openpi；完整数据与训练管线未全部公开） |
| 原文 | https://arxiv.org/abs/2504.16054 |

## 核心原理

π0.5不是简单把π0换一批数据继续训练。它面向家庭移动操作的长时任务，把统一模型拆成两个训练阶段和两种推理节奏：预训练用离散表示吃下异构数据，后训练再为目标本体加入连续动作专家。执行时同一模型先产生语义子任务，然后以它为条件生成低层动作。

### 流程直觉

```mermaid
flowchart LR
  A["问题 / 数据 / 观测"] --> B["π0.5"]
  B --> C["控制 / 策略 / 数据产物"]
  C --> D["评测或真机闭环"]
```

模块边界与符号定义以原文为准；上图只固定阅读骨架。

## 工程实践

FAST token保留一段动作的时序结构，使动作数据可以和“下一步子任务是什么”这类文本监督共用自回归训练接口；本体、相机和语言仍作为条件。后训练时连续flow expert重新接管精细动作输出，因此离散预训练表示负责迁移语义，连续头负责目标机器人控制。两阶段使用的动作表示不同，必须有明确的本体适配和对齐，不能把FAST token直接当成电机命令。

| 检查项 | 建议 |
|--------|------|
| 一手来源 | 回 arXiv / DOI / 项目页核对数值与声明 |
| 开源边界 | 部分开源（openpi；完整数据与训练管线未全部公开） |
| 本库定位 | 详情编译页；深入公式与实验表读原文 |

## 源码运行时序图

**不适用**（部分开源（openpi；完整数据与训练管线未全部公开））。若后续官方发布可运行训练/推理入口，应补 `sources/repos/` 并更新本图。

## 实验与评测读法

- 把「仿真指标 / 真机证据 / 仅项目演示」分开记账。
- 对照同组相邻工作（见关联页面）时，对齐任务定义与观测接口，再比成功率。
- 综述类条目关注分类框架与缺口，不把引用列表当作选型排名。

## 结论

**π0.5 应作为 HMI「世界模型、VLA与Agent」线上的独立知识节点阅读：先抓住其真正改变的问题接口，再决定是否进入复现或对比实验。**

- 核心贡献是问题表达或管线接口，而不只是单一网络结构名。
- 开源状态：部分开源（openpi；完整数据与训练管线未全部公开）。
- 与本库已有相邻页交叉阅读，避免重复造页。
- 数值、消融与许可以一手来源为准；本页是编译索引。
- 若官方后续补齐代码/数据，应回写 `sources/` 与本节开源字段。

## 局限与风险

- 后训练集中使用与目标家庭机器人最相关的数据，加入flow-matching action expert，并包含人类监督者给出的高层子任务。运行时，模型以较慢节奏输出“走到水槽前”、“拿起海绵”这类语义动作，再以当前子任务为条件高频生成action chunk。这是模型内的分层控制：两层共享表示，但不用同一频率重做全部决策。
- 勿把 HMI 解读中的工程判断直接写成论文作者承诺。
- 经典控制论文与现代 RL/VLA 论文的「可复现」标准不同，选型时分开评估。

## 与其他工作对比

| 维度 | 本工作（π0.5） | [π0](../methods/π0-policy.md) | [π0.7](../methods/pi07-policy.md) | [基础策略](../concepts/foundation-policy.md) |
|------|----------------|------------------------------|----------------------------------|----------------------------------------------|
| 动作表示 | 预训 FAST 离散 + 后训连续 flow 专家 | 单一 flow-matching 连续动作 | 多模态提示 steer，异质数据对齐 | 通才/基础策略的抽象概念 |
| 目标场景 | 家庭移动操作长时任务、开放世界泛化 | 复杂操作的通用建模 | 组合任务 + 跨本体泛化 | 单一大模型覆盖多任务/本体 |
| 分层 | 先出语义子任务再高频 action chunk | 无显式两阶段分层 | 训练/推理提示 steer | — |
| 关系/取舍 | π0 的后继，加两阶段+分层；两种动作表示须对齐 | π0.5 的基座前身 | π 系更新代，蒸馏 RL 专精 | π 系均属其一种实现 |

## 关联页面

- [HMI 论文覆盖导读](../queries/hmi-papers-coverage.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [π0-policy](../methods/π0-policy.md)
- [pi07-policy](../methods/pi07-policy.md)
- [vla](../methods/vla.md)
- [Indi](./paper-indi.md) — 在 π0.5 与 GR00T-N1.7 上蒸馏行为意图（arXiv:2608.23478）
- [foundation-policy](../concepts/foundation-policy.md)
- [Emergent Transfer](./paper-emergent-transfer-cross-config.md) — 以 π₀.₅ 为骨干的跨配置遗留数据三相共训研究
- [RoboHarness](./paper-robo-harness.md) — 以 π₀.₅ 为异构策略库成员之一的长时程编排框架
- [ActFovea](./paper-actfovea.md) — 对冻结 π₀ 加运行时防护（π 系列部署侧对照）
- [WCM](./paper-wcm-world-critic-model.md) — 把 π₀.₅ 作为 flow matching 主干做 RL 后训练
- [CLIFT](./paper-clift-closed-loop-iterative-finetuning.md) — 以 π₀.₅ 作开放权重对照，同管线下明显落后于托管 API 的 GROD
- [G0.5](./paper-galaxea-g05.md) — VLM-as-Actor 开源对照；真机微调表直接打 π₀.₅
- [χ₀ / kai0](./paper-kai0.md) — 以 π₀.₅ 为唯一可跑通骨干做协同叠衣后训练；相对基线 SR 约 +250%
- [SPD](./paper-spd.md) — 灵巧真机：π0 风格单帧长 chunk 弱于历史+短 chunk（CoRL 2026）
- [FlashVLA](./paper-flashvla.md) — 在 \(\pi_{0.5}\) 上做流式异步动作解码（arXiv:2608.27384，已开源）
- [CLAP](./paper-clap-cross-embodiment.md) — 跨本体视频 WM 对 \(\pi_{0.5}\) 做推理时规划（arXiv:2608.27406，已开源）

## 参考来源

- [sources/papers/hmi_p059_pi05-open-world-vla.md](../../sources/papers/hmi_p059_pi05-open-world-vla.md)
- [SPD 论文归档](../../sources/papers/spd_corl_2026.md) — 灵巧真机上 π0 风格 chunk 对照
- [sources/repos/humanoid-motion-intelligence.md](../../sources/repos/humanoid-motion-intelligence.md)
- [HMI 论文总索引](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/README.md)
- [sources/papers/chi0_kai0_arxiv_2602_09021.md](../../sources/papers/chi0_kai0_arxiv_2602_09021.md) — χ₀ 以 π₀.₅ 为生产向后训练基线

## 推荐继续阅读

- [arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
- [项目/官方解读](https://www.physicalintelligence.company/blog/pi05)
- [代码](https://github.com/Physical-Intelligence/openpi)
- [HMI 逐篇解读 P059](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P059.md)

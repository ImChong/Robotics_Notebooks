---
type: entity
project_id: esp-energy-score-policy
tags: [paper, manipulation, vla, flow-matching, one-step-policy, imitation-learning, pi05, gr00t, franka]
status: complete
updated: 2026-10-12
arxiv: "2610.07696"
project: https://lilika-makabe.github.io/energy-score-policy-site/
venue: "arXiv 2026"
summary: "ESP：用能量分数（严格正当评分规则）直接训练 VLA 动作头，一次网络评估输出多模态动作块；LIBERO 平均 0.918 优于 10 步流匹配的 0.891，动作专家快约 8.5 倍，Franka 真机完整 Pick & Place 7/15 对 1/15。"
related:
  - ../overview/frontier-manipulation-2026-10-05-10-09.md
  - ../tasks/manipulation.md
  - ../concepts/flow-matching-embodied-policy.md
  - ../methods/diffusion-policy.md
  - ../methods/action-chunking.md
  - ./paper-pi05-open-world-vla.md
  - ./paper-gr00t-n1-5.md
  - ./paper-immiscible-diffusion-policy.md
sources:
  - ../../sources/papers/esp_energy_score_policy_arxiv_2610_07696.md
  - ../../sources/sites/esp-energy-score-policy-project.md
  - ../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md
---

# ESP（Energy-Score Policy）

**ESP**（arXiv:2610.07696，Makabe / Kim / Matsushita）是一种**无教师的一步多模态动作生成**方法：把策略上下文与噪声 z ~ N(0, I) 一次映射为动作块，用**能量分数**而不是 MSE 训练动作头，使单次前向既保留多模态又省掉流匹配 / 扩散的迭代去噪。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ESP | Energy-Score Policy | 本文方法，能量分数训练的一步动作头 |
| NFE | Number of Function Evaluations | 生成一个动作块所需的网络评估次数，ESP 为 1 |
| VLA | Vision-Language-Action model | 视觉语言动作模型，本文接在 π0.5 与 GR00T 上 |
| XM | hard-min over K candidates | 对 K 个候选取最近者的一步基线（论文对照） |
| LIBERO | LIBERO benchmark | 终身机器人学习仿真基准，含 Spatial/Object/Goal/Long10 |
| FFN | Feed-Forward Network | 动作专家中的前馈层，论文测试其加宽 |

## 为什么重要

- **延迟**：VLA 的流匹配动作头每个 chunk 需 10 次（π0.5）或 4 次（GR00T）评估；ESP 把动作专家从 28.39 ms 降到 3.35 ms。
- **理论上不丢模态**：能量分数严格正当，期望仅在真实分布处最小，MSE 则收敛到条件均值（论文 Proposition 1）。
- **可直接换头**：冻结视觉语言骨干，只替换/微调动作头。

## 方法栈与流程总览

```mermaid
flowchart LR
  O["图像+指令+本体 (冻结 VLM 骨干)"] --> C["上下文特征"]
  Z["噪声 z ~ N(0,I)"] --> H["一步 ESP 动作头 (t 固定为 1)"]
  C --> H
  H --> A["动作块 (NFE=1)"]
  A -.训练.-> L["K=8 个候选: 能量分数损失"]
```

## 方法

- **损失**：ES(Q, y) = E‖x−y‖ − ½E‖x−x′‖（β = 1）；有限样本估计用每个训练样本 K = 8 个候选，第一项拉近候选与真值，第二项（两两距离，O(K²)）鼓励候选分散。
- **π0.5 + LIBERO**：冻结 VLM 骨干，流匹配头换成随机初始化的一步 ESP 头；批 32、AdamW、EMA 0.999、预热 10k 步到 5e-5，训练 60k 步、3 个 seed，预测 10 步执行 5 步。
- **GR00T-N1.7-3B + 真机**：冻结 VLM，仅微调预训练动作头；批 8、lr 2.5e-5；Drawer Closing 60k 步、Pick & Place 100k 步，预测并执行 8 步。
- **对照**：XM（NFE=1）、原生流匹配（π0.5 NFE=10；GR00T NFE=4）、MSE。

## 实验与评测

- **TwoBranch 双模态合成**（单 seed）Match rate：MSE 0.7%，一步流 4.2%，XM 91.9%，ESP 96.2%，10 步流 97.1%。
- **LIBERO（π0.5，3 seed）**：ESP 平均 0.918 ± 0.014，Flow 0.891，XM 0.866；Long10 上 0.813 / 0.802 / 0.672。
- **延迟（A100，batch 1）**：动作专家 3.35 vs 28.39 ms（8.47×）；缓存端到端 4.46 vs 29.14 ms（6.53×）；未缓存端到端 59.01 vs 82.54 ms（仅 1.40×，被 VLM 主干占据）。
- **加宽动作专家**（FFN 73728，总 7.204B）：平均 0.937，21.15 ms。
- **多模态消融**（6 个 LIBERO-10 任务 × 32 rollout）：成功 ESP 162、XM 127、Flow 155。
- **真机（Franka Research 3，GR00T，每项 15 次）**：Drawer Closing 12/15 vs Flow 8/15；只抓取 10/15 vs 9/15；完整 Pick & Place 7/15 vs 1/15。
- **训练成本**：ESP 15k 步 / 4.78 h 达 93.0%，Flow 60k 步 / 4.98 h 达 93.5%，但单步 Flow 比 ESP 快 3.81×。

## 源码运行时序图

不适用：截至 2026-10-12 论文与项目页均未提供代码仓库（项目页无 “coming soon”）。训练接口可对照 openpi / GR00T 的动作头替换点自行实现（推测）。

## 工程实践

- 推理侧：一次采样噪声、一次前向；注意端到端收益取决于 VLM 骨干是否可缓存（缓存 6.53×，未缓存 1.40×）。
- 训练侧：K=8 候选带来 O(K²) 项和激活内存，单步慢 3.81×，但步数少得多，总墙钟相当。
- 开源状态：**未发布**；项目页的 Real-world Results 小节在 2026-10-12 抓取时为空。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [π0.5](./paper-pi05-open-world-vla.md) | 本文替换其流匹配动作头，保留骨干 |
| [GR00T N1.5](./paper-gr00t-n1-5.md) | 论文使用 GR00T-N1.7-3B 作真机骨干，N1.5 为同系列早期页 |
| [Immiscible DP](./paper-immiscible-diffusion-policy.md) | 同为保多模态，但后者仍多步扩散，只改噪声配对 |
| [Flow matching 概念页](../concepts/flow-matching-embodied-policy.md) | ESP 是其“迭代采样”假设的替代 |

## 结论

**ESP 把“多模态”从迭代采样里解耦出来：真正受益的是动作头延迟，而不是端到端延迟，成功率与流匹配相当甚至略高。**

1. **看动作专家延迟 vs 端到端延迟** — 8.47× 对 1.40×，差距来自 VLM 骨干。
2. **K 是主要超参** — 论文固定 K=8，未做 K 消融。
3. **比 XM 稳** — LIBERO Long10 0.813 对 0.672。
4. **真机样本小** — 每项 15 次，7/15 vs 1/15 需更多 trial 验证。
5. **训练略贵** — 单步 3.81×，但步数更少。

## 局限与风险

- 单 seed 的 TwoBranch 只作示意；真机每项 15 次。
- 训练每步算力更高；降低训练成本留作未来工作（论文自述）。
- 无代码，无作者单位信息（未核实）；复现门槛高于“一行换头”的宣传观感。

## 关联页面

- [Manipulation 周报汇总（2026-10-05 至 10-09）](../overview/frontier-manipulation-2026-10-05-10-09.md)
- [Flow Matching 具身策略](../concepts/flow-matching-embodied-policy.md)
- [Diffusion Policy 方法页](../methods/diffusion-policy.md)
- [Manipulation 任务页](../tasks/manipulation.md)

## 参考来源

- [esp_energy_score_policy_arxiv_2610_07696.md](../../sources/papers/esp_energy_score_policy_arxiv_2610_07696.md)
- [esp-energy-score-policy-project.md](../../sources/sites/esp-energy-score-policy-project.md) — 项目页核查（无代码）
- [frontier_manipulation_2026_10_05_10_09.md](../../sources/blogs/frontier_manipulation_2026_10_05_10_09.md) — 周报线索
- 论文：<https://arxiv.org/abs/2610.07696>

## 推荐继续阅读

- [项目页](https://lilika-makabe.github.io/energy-score-policy-site/)
- [Gneiting & Raftery：Strictly Proper Scoring Rules](https://doi.org/10.1198/016214506000001437)

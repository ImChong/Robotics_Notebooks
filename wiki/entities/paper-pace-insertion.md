---
type: entity
tags: [paper, tactile, visuotactile, contact-rich, manipulation, insertion, action-chunking, xiaomi]
status: complete
updated: 2026-10-26
project_id: pace-insertion
arxiv: "2610.16308"
project: https://pace-insertion.github.io/
institutions: [xiaomi]
related:
  - ../concepts/tactile-sensing.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/visuo-tactile-fusion.md
  - ../methods/diffusion-policy.md
  - ./paper-foretac.md
sources:
  - ../../sources/papers/pace_arxiv_2610_16308.md
  - ../../sources/sites/pace-insertion.md
summary: "PACE（arXiv:2610.16308）将触觉分作持续接触状态、突发接触事件与未来接触后果三种时间角色，在五个真机插入任务平均成功率 90.7%；截至 2026-10-26，代码、模型和数据仍未公开。"
---

# PACE：让触觉承担三种时间角色

**PACE（Phase-Aware Contact-Event Conditioning）把触觉流拆成状态条件、事件纠正和后果预测三条互补路径：状态稳定动作块生成，事件修正动作块，预测目标则在训练中塑造动作表示。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PACE | Phase-Aware Contact-Event Conditioning | 本文的触觉时间角色条件化框架 |
| CoRL | Conference on Robot Learning | 项目页标注的 2026 年会议发表信息 |
| DP | Diffusion Policy | PACE 跨架构实验接入的扩散策略骨干 |
| GF225 | GF225 tactile sensor | PACE 默认使用的触觉信号源之一 |
| ATB | Assembly Task Board | NIST 装配任务板，供 ManipulationNet 插入测试使用 |

## 为什么重要

接触丰富插入中，视觉负责粗对位，但隐藏插针、微小间隙与突然卡滞需要触觉反馈。将整条触觉流简单拼接到策略输入，会把持续状态、瞬时事件和未来接触影响混在一个融合通路里。PACE 按时间作用拆分通路，同时保留动作块的连续生成，并给突发接触单独的纠正入口。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | Xiaomi Robotics（小米机器人） |
| **作者** | Zhiyuan Chen、Shixiong Xu、Shuai Chen、Yangwei You |
| **发表** | CoRL 2026 Oral；项目页称 Top 1% |
| **任务** | Key、Two-Pin、Three-Pin、USB 与 Ethernet 五类真机插入 |
| **示范采集** | Force Dimension sigma.7 力反馈遥操作；每任务约 160 条、每条约 300 步 |
| **开源** | 代码、模型与数据集标为将发布；截至 2026-10-26 未列公开链接 |

## 核心原理

### 三条触觉路径

1. **状态条件（Phase）：** 将持续接触状态送入 query-time 特征调制，影响动作块生成。
2. **事件纠正（Event）：** 对碰撞等突发触觉事件建立独立路径，修正当前生成的动作块。
3. **后果预测（Prediction）：** 训练中预测接触后果并监督 decoder hidden states，让动作表示在执行前包含接触信息。

三条路径消费共享触觉流，但承担不同的时序功能。前两条参与在线动作生成与纠正，后果预测作为训练监督塑形策略表征。

## 实验与评测

| 评测 | 结果 | 读法 |
|------|------|------|
| 五项真机插入任务 | 平均成功率 **90.7%** | 相对 vision-only 和普通视触觉融合基线提升超过 44 个百分点 |
| ManipulationNet / NIST ATB | 六种 peg-hole 几何与间隙配置平均 **88.7%** | 测试包括 3 mm、1 mm 与 0.1 mm 间隙 |
| 跨架构 | DP+PACE 在 2-pin 与 Key 上平均 **88.3%** | 较 vanilla DP 高 70.0 个百分点 |
| 跨反馈信号 | 末端力/力矩 90.0%；关节电流 88.3% | 时间角色设计不局限于默认触觉传感器 |

五项任务针对不同接触难点：非轴对称方向约束、视觉不可见双插针、三点几何约束、小间隙高位姿敏感度，以及有弹性卡扣的持续施力。以上结果受具体装配件、示范和评测设置约束。

## 与其他工作对比

| 工作 | 触觉如何进入策略 | 主要应用 |
|------|-----------------|----------|
| **PACE** | 状态条件 + 事件纠正 + 训练期后果预测 | chunk-based 精细插入；触觉角色按时间功能拆开 |
| [ForeTac](./paper-foretac.md) | 预测候选动作未来触觉，用接触质量梯度在生成时引导动作 | 擦拭、刷卡、易碎物抓取；强调候选动作后果优化 |
| 普通视触觉融合 | 将视觉与当前触觉在策略输入或生成期融合 | 提供当前多模态观测，未必显式区分状态、事件和后果 |

PACE 的可迁移证据来自将框架用于 Diffusion Policy 与替代物理反馈信号；ForeTac 的主要增量则是冻结基座策略下的动作生成期梯度引导。二者解决相近接触问题，机制和任务并不相同。

## 工程实践

- 设计触觉通路时先区分稳态信息与突发事件，避免一个融合模块同时承担慢条件与快纠偏。
- 后果预测可作为训练信号塑造表示；部署时要明确其作用是否仅训练期，避免把辅助头误放进实时控制预算。
- 力反馈遥操作有助于获得碰撞退让、搜索调整和稳定插入示范；每任务约 160 条、约 300 步可作数据规模参考，不能直接当作最低需求。
- 0.1 mm 间隙结果来自特定标准件与测试台，应与真实产线公差、夹具、传感器分辨率和控制带宽分开验证。

## 源码运行时序图

**不适用：** 截至 2026-10-26，项目页未提供可运行代码；页面称代码、模型和数据集将发布。

## 结论

**PACE 的主要贡献是把触觉按时间角色分工，使动作块既能利用持续接触状态，又能对瞬时接触事件纠偏，并在训练阶段学习接触后果。**

1. **结果针对接触插入。** 五项真机任务平均 90.7%，不能外推为通用操作成功率。
2. **三条路径互补。** 消融应分别查看状态条件、事件纠正与后果预测的贡献。
3. **性能要绑定约束。** 88.7% 覆盖六种 ATB 几何-间隙组合，包含 0.1 mm，不代表任意公差装配。
4. **跨信号与跨骨干是积极证据。** 已报告的替代信号和 Diffusion Policy 迁移任务数仍有限。
5. **目前不能复现。** 项目页称代码、模型与数据后续开放，公开后再核对运行时细节。

## 局限与风险

- **代码与数据未公开：** 截至 2026-10-26，项目页未列 GitHub、模型或数据集下载链接。
- **任务覆盖有限：** 五个插入任务与一个标准装配板不足以证明泛化到其他机器人、夹具、材料或全身接触操作。
- **纠正带宽待核：** 触发条件、频率与完整实时延迟需结合实现验证，不能仅凭成功率判断部署安全性。
- **设备依赖：** 示范使用 sigma.7 力反馈遥操作，采集质量和成本可能因示范接口而异。

## 关联页面

- [触觉感知](../concepts/tactile-sensing.md) — 触觉信号与传感方式
- [接触丰富操作](../concepts/contact-rich-manipulation.md) — 相关任务族
- [视触觉融合](../concepts/visuo-tactile-fusion.md) — 跨模态输入组织
- [ForeTac](./paper-foretac.md) — 预测接触后果并引导生成策略的相邻工作

## 参考来源

- [PACE 论文归档](../../sources/papers/pace_arxiv_2610_16308.md)
- [PACE 项目页归档](../../sources/sites/pace-insertion.md)
- [PACE arXiv](https://arxiv.org/abs/2610.16308)

## 推荐继续阅读

- [PACE 项目页](https://pace-insertion.github.io/)
- [ForeTac](./paper-foretac.md)

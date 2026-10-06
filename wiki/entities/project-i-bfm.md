---
type: entity
tags: [project, humanoid, behavioral-foundation-model, loco-manipulation, object-interaction, mujoco-demo, roboparty]
status: complete
updated: 2026-10-06
paper: "2610.06129"
url: https://iamhardworking.github.io/I-BFM/
related:
  - ./paper-i-bfm.md
  - ./paper-bfm-zero.md
  - ../concepts/behavior-foundation-model.md
sources:
  - ../../sources/sites/i-bfm-project-page.md
  - ../../sources/papers/i_bfm_arxiv_2610_06129.md
summary: "I-BFM 官方项目展示站：提供论文、概览/Carry-Push-Kick/目标到达视频和浏览器内 MuJoCo Carry Box/键盘控制入口；结果卡按 300 episodes 展示三类交互与两种扰动成功率。页面标注代码 Coming soon；网站若干 SR 与 arXiv Table I 不一致，分别保留来源。"
---

# I-BFM 项目页：Reward-Conditioned Humanoid Interaction

**类型：** 人形–物体交互行为基础模型研究项目及官方演示页  
**配套论文：** [I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning](./paper-i-bfm.md) · [arXiv:2610.06129](https://arxiv.org/abs/2610.06129)  
**官方项目页：** <https://iamhardworking.github.io/I-BFM/>  
**作者团队：** Tongji / Tsinghua / Zhejiang / RoboParty Lab / HIT / ShanghaiTech  
**状态：** 项目页面公开；页面标注 **Code · Coming soon**。截至 2026-10-06 未发现公开代码仓库、checkpoint 或数据下载入口。

## 项目页提供什么

该站点是论文配套的视觉演示与试用入口，不是一个已发布的训练代码仓库。首页提供：

- **研究概览视频**：呈现 I-BFM 预训练及奖励推断的框架。
- **浏览器内交互演示**：Live MuJoCo Carry Box 与键盘控制入口。该交互 demo 可用于体验项目所表达的闭环操作概念，但不能据此复现训练。
- **鲁棒性视频**：Carry Box、Push Box、Kick Box 的受扰表现，以及 Robust Goal Reaching。
- **结果卡片**：把 Carry / Push / Kick 在无扰动、物体受扰、机器人受扰下的 300 episode 计数放到同一页面。

## 项目站结果卡（按页面原值记录）

| 条件 | Carry Box | Push Box | Kick Box |
|---|---:|---:|---:|
| 无外部扰动 | 94.33% (283/300) | 90.00% (270/300) | 81.67% (245/300) |
| 物体受扰 | 91.00% (273/300) | 85.33% (256/300) | 79.67% (239/300) |
| 机器人受扰 | 89.33% (268/300) | 84.67% (254/300) | 86.67% (260/300) |

### 与论文表格核对

网站结果不完全等于 arXiv v1 Table I：例如 Push 标称成功率网页为 90.00%，论文表格为 75.0 ± 6.6%；Kick 标称网页为 81.67%，论文表格为 70.0 ± 3.6%。机器人扰动下 Kick 网页为 86.67%，论文表格为 60.0 ± 8.7%。项目页没有解释版本或协议差异。因此：

1. 本项目页只把上述数字标为**网站结果卡所示**；
2. 论文节点保留 arXiv 表格数据与论文正文中 Carry Eobj 的数值冲突；
3. 在作者澄清前，不推断哪一套数字更新，也不把项目站的 300-episode 数值写成论文表格结果。

## 方法归属与论文边界

项目页展示的核心是论文提出的 I-BFM，而不是单独的另一个算法：FB 表征与无监督 RL 学习统一 latent 行为，物体/接触观测形成闭环交互上下文；LOGO 将近端局部交互意图与远端目标分别条件化。Carry / Push / Kick、Goal Reaching、style control 和 skill chaining 的技术设计、消融与实验定义见[独立论文详情](./paper-i-bfm.md)。

真机视频使用 Unitree G1 与动捕工作区，论文报告 0.35 m、0.7 kg 箱体和 50 Hz 执行；这属于定性硬件验证，不是网站仿真结果卡所列的统计评测。

## 复现与资源状态

| 资源 | 当前状态 |
|---|---|
| 项目页与演示视频 | 已公开 |
| 浏览器内 MuJoCo 交互入口 | 项目页嵌入/链接提供 |
| 论文 | [arXiv:2610.06129](https://arxiv.org/abs/2610.06129) |
| 代码仓库 | 页面标注 Coming soon；未找到公开仓库 |
| 权重 / 训练数据下载 | 未发现公开入口；论文也未给出可下载链接 |

## 结论

I-BFM 项目页的独立价值在于把论文中的统一交互策略做成可视化展示，并提供轻量 MuJoCo 交互入口。当前读者可以观察 Carry、Push、Kick、恢复和长程串联的目标行为；但网站的仿真结果卡与论文表格有明显 SR 差异，使用时应明确标注来源。代码仍为 Coming soon，demo 不能替代训练实现的公开。

## 关联页面

- [论文详情：paper-i-bfm](./paper-i-bfm.md)
- [BFM-Zero](./paper-bfm-zero.md) — I-BFM 的身体优先 FB-BFM 相关工作
- [行为基础模型概念](../concepts/behavior-foundation-model.md)

## 参考来源

- [官方 I-BFM 项目页](https://iamhardworking.github.io/I-BFM/)
- [i-bfm-project-page.md](../../sources/sites/i-bfm-project-page.md)
- [配套论文来源归档](../../sources/papers/i_bfm_arxiv_2610_06129.md)
- [arXiv:2610.06129](https://arxiv.org/abs/2610.06129)

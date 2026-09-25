---
type: overview
tags: [overview, survey, vla, humanoid, manipulation, technology-map]
status: complete
updated: 2026-09-24
related:
  - ../entities/paper-nowwam.md
  - ../entities/paper-tandem.md
  - ../entities/paper-forgetmimic.md
  - ../entities/paper-infinova.md
  - ../entities/paper-dissect-vla-post-training.md
  - ../entities/paper-amplify-robotics.md
  - ../entities/paper-brickcraft-duo.md
  - ../entities/paper-plantorv.md
  - ../entities/paper-davis-humanoid-soccer.md
  - ../entities/paper-stein-admm-contact.md
  - ../entities/paper-mrsvlmra.md
  - ../entities/paper-remote-surfaces-electrovibration.md
  - ../entities/paper-robot-group-joining.md
  - ../methods/vla.md
  - ../methods/reinforcement-learning.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md
summary: "具身智能小站 2026-09-24 十三篇盘点：生成式控制适配、按需示范、motion unlearning 与视角/接触/多机安全五条阅读线。"
---

# 具身智能十三篇：生成式控制、遗忘与安全更新

> **本页定位**：为 [具身智能小站 · 13 篇盘点](https://mp.weixin.qq.com/s/4QpQgKEw7BnLFzhG-05uzg)（2026-09-24）提供按问题组织的阅读坐标。

## 一句话观点

**生成式先验、示范预算、后训练诊断与人形安全更新应分开优化——ForgetMimic 的「会忘记」与 NowWAM 的「不必预测未来」是同一时代的两种控制接口问题。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| TAMP | Task and Motion Planning | 任务与运动规划 |
| WAM | World Action Model | 世界–动作联合模型 |
| ADMM | Alternating Direction Method of Multipliers | 交替方向乘子法 |

## 为什么单独做这张地图

- 一次列出 13 篇跨度大的工作；需要横切面索引。
- **13 篇各有一页**，可逐篇点开核对开源与评测口径。

## 阅读优先级（文内三档）

1. **深读：** [NowWAM](../entities/paper-nowwam.md) — 生成式去噪接口
2. **跟进：** [TANDEM](../entities/paper-tandem.md)、[ForgetMimic](../entities/paper-forgetmimic.md)
3. **扫读：** 其余 10 篇按研究方向收藏

## 节点索引

| # | 短名 | 详情页 | 一句话 |
|---|------|--------|--------|
| 01 | NowWAM | [nowwam](../entities/paper-nowwam.md) | 生成式视觉先验不必预测未来画面；沿完整去噪轨迹适配当前观测即可稳定控制，并显著降 token 与步时。… |
| 02 | TANDEM | [tandem](../entities/paper-tandem.md) | 规划器能走的步骤交给 TAMP，只在能力缺口处按需遥操作，并把各阶段拼成完整 VLA 微调示范。… |
| 03 | ForgetMimic | [forgetmimic](../entities/paper-forgetmimic.md) | 多技能人形策略可在 **动作级** 选择性遗忘指定 motion，而不必整策略重训。… |
| 04 | InfiNoVA | [infinova](../entities/paper-infinova.md) | 用时变 3D Gaussian 重建轨迹并渲染无限新视角，提升 VLA 对 **未见相机位姿** 的鲁棒性。… |
| 05 | DissectVLA | [dissect-vla-post-training](../entities/paper-dissect-vla-post-training.md) | 把 VLA 优势引导后训练拆成 **构造 / 校准 / 利用** 三阶段，用离线诊断筛设计再少跑真机。… |
| 06 | Amplify | [amplify-robotics](../entities/paper-amplify-robotics.md) | 用声明式 AMPL 模型表达机器人 NLP 问题，核心库 **537 行**，便于跨求解器复现轨迹优化基准。… |
| 07 | BrickCraft-Duo | [brickcraft-duo](../entities/paper-brickcraft-duo.md) | 互锁积木双臂装配：可复用单/双臂技能 + 稳定性感知组合 + 人机定向修正，最长 **9 步** 长时任务。… |
| 08 | PLANTORV | [plantorv](../entities/paper-plantorv.md) | VLM 擅长语义描述但不等于可靠几何；框架把 VLM 标注与 RGB-D 几何拆开再合成对象级表示。… |
| 09 | DAVIS | [davis-humanoid-soccer](../entities/paper-davis-humanoid-soccer.md) | **168×80 深度-only + 主动头**；可见性门控几何 + GT→prediction annealing；射门/带球分 checkpoint；Noetix E1 真机（[项目页](https://thusi-lab.github.io/DAVIS/)） |
| 10 | Stein-ADMM | [stein-admm-contact](../entities/paper-stein-admm-contact.md) | 接触隐式 TO 易陷单一局部接触模式；Stein 排斥力加在 ADMM 分裂变量上可发现 **多样** 抓取/推/交接策… |
| 11 | MRSVLMRA | [mrsvlmra](../entities/paper-mrsvlmra.md) | 感知不对称多机协作：有相机四足共享语义场景，LLM 分工，zonotope 可达性门拦截不安全语言建议。… |
| 12 | Remote Surfaces | [remote-surfaces-electrovibration](../entities/paper-remote-surfaces-electrovibration.md) | 电振动触觉把远程刚性接触映射到触摸屏，改善遥操作响应时间与临场感（N=21 用户研究）。… |
| 13 | Robot Group Joining | [robot-group-joining](../entities/paper-robot-group-joining.md) | 语言引导预测「社会上合适的加入站位」，而非仅几何路径到固定目标点。… |

## 关联页面

- [VLA 方法页](../methods/vla.md)
- [强化学习](../methods/reinforcement-learning.md)
- [操作任务](../tasks/manipulation.md)
- [ForgetMimic 实体](../entities/paper-forgetmimic.md)

## 参考来源

- [13 篇盘点（公众号）](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [Universal Post-Training](../concepts/universal-post-training-robotics.md)

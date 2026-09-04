# SmoothRL（arXiv:2608.29768）

> 来源归档（ingest）

- **标题：** SmoothRL: Online Reinforcement Learning During Asynchronous Execution
- **类型：** paper / vla / online-rl / action-chunking / asynchronous-inference / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2608.29768>
- **PDF：** <https://arxiv.org/pdf/2608.29768>
- **HTML：** <https://arxiv.org/html/2608.29768>
- **项目页：** <https://www.astribot.com/research/SmoothRL>（2026-09-02 入库时 **404**；**2026-09-04 再核已上线**，见 [站点归档](../sites/astribot-smoothrl.md)）
- **机构：** 星尘智能（Astribot）
- **作者：** Guang Gao\*、Yuxuan Nong\*、Baifu Huang；Project Lead：Jianan Wang（Astribot Team）
- **发表 / 上传：** 2026-08（arXiv）
- **平台：** Astribot S1；冻结 π₀.₅ base VLA
- **入库日期：** 2026-09-02；再核 2026-09-04

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.29768](https://arxiv.org/abs/2608.29768) | 论文主文 |
| 项目页 | [astribot.com/research/SmoothRL](https://www.astribot.com/research/SmoothRL) | 2026-09-04 上线；hero + 9 段任务对比视频 |
| 对照 | [ARLI arXiv:2608.23831](https://arxiv.org/abs/2608.23831) | 异步 VLA + RL（DSRL 噪声舵，非 value-gradient 穿 raw action） |
| 骨架 | RLT（arXiv:2604.23073） | 冻结 VLA + RL token + 残差 actor；本文沿用、未单独升格 |
| 方法 | [action-chunking](../../wiki/methods/action-chunking.md) | chunk 级 MDP 与异步执行 |
| 同期公众号 | [深蓝六条路线](../blogs/wechat_shenlan_embodied_six_routes_holes_2026-09-04.md) | 把 RL 读成「VLA 后训练」；非 SmoothRL 通稿 |

## 开源状态（步骤 2.5，2026-09-04 复核）

- 项目页 **已上线**，仅 arXiv + OSS 演示视频。
- arXiv / 项目页 **未列** GitHub / Hugging Face / 权重。
- **处理：** wiki 标 **确认未开源**；`## 源码运行时序图` 标不适用。

## 摘要级要点

- **问题：** VLA/WAM 部署需 **可靠性（在线 RL）** 与 **平滑实时（异步 chunk）** 兼得；同步假设的在线 RL 与部署动力学不匹配。
- **SmoothRL：** value-gradient；chunk 分 **committed / execution / discarded**；**∇ₐQ 仅经 execution region [n,2n)**；训练 rollout 嵌入与部署相同的异步环。
- **实例化：** 冻结 π₀.₅ + 3×512 MLP actor/critic 改 20 维臂；TT-RTC；REDQ 风格 critic 集成；actor 损失 = Q + BC + 速度/加速度/jerk 平滑项（式 5）；correction bound 0.05。
- **干预：** 绝对（VR）/ 残差（摇杆）；远距投掷 VR ~30% vs 残差 ~80%。
- **真机三任务（250 episodes）：** 投掷 39%→**94%**；笔帽 8%→**83%**；开箱 30%→**90%**（开箱 150 ep 曾掉到 20%）。
- **运动质量：** 投掷末端 RMS 加速度 −52%、jerk −47%。
- **局限（§5）：** 固定 latency budget；残差表达力受 base 与 0.05 邻域限制。
- **未来：** 梯度可穿整个生成式策略（QAM 式）；把 blending 写进目标；更大任务分布。

## 核心摘录（面向 wiki 编译）

### 1) 异步环与 latency budget（§3.1）

- 推理 5 Hz、控制 30 Hz → **n=6** 帧；H=32 → committed [0,6)、execution [6,12)、discarded [12,32)。
- 固定 budget：每 chunk 提前 n 步请求；推理提前完成则等待。

### 2) 平台与实现（§3.4 / §4.1）

- S1：25 DoF；三相机 224²；base 31-dim；RL 只改 20 维臂。
- Actor/critic：3 层 MLP、512、LayerNorm；batch 256；G=5、D=5。
- 50 ep 冻结 base 预热 buffer；稀疏终端奖励由操作员判定。

### 3) 结果（Table 1）

| Task | Base | 150 ep | 200 ep | 250 ep |
|------|------|--------|--------|--------|
| Dynamic Tossing | 39% | 72% | 83% | **94%** |
| Pen Capping | 8% | 67% | 75% | **83%** |
| Box Opening | 30% | 20% | 40% | **90%** |

### 4) 干预模式（§4.3）

远距投掷：VR 直接遥操作 chunk ~30% vs **残差修正 ~80%**。

## 对 wiki 的映射

- 沉淀实体页：[SmoothRL](../../wiki/entities/paper-smoothrl.md)
- 交叉补强：[ARLI](../../wiki/entities/paper-arli.md)、[VLA](../../wiki/methods/vla.md)、[action-chunking](../../wiki/methods/action-chunking.md)、[Lumo-2](../../wiki/entities/lumo-2.md)、[Philia](../../wiki/entities/philia.md)
- 同期公众号写回：[六条路线的窟窿](../../wiki/queries/embodied-six-routes-holes.md)

## 当前提炼状态

- [x] arXiv HTML §3–5 / Table 1 / Contributions 摘录
- [x] 项目页 2026-09-04 再核：已上线、无代码、9+1 视频
- [x] 升格 / 更新 `wiki/entities/paper-smoothrl.md`

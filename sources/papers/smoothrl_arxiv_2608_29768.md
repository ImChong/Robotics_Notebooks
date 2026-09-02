# SmoothRL（arXiv:2608.29768）

> 来源归档（ingest）

- **标题：** SmoothRL: Online Reinforcement Learning During Asynchronous Execution
- **类型：** paper / vla / online-rl / action-chunking / asynchronous-inference / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2608.29768>
- **PDF：** <https://arxiv.org/pdf/2608.29768>
- **HTML：** <https://arxiv.org/html/2608.29768>
- **项目页：** <https://www.astribot.com/research/SmoothRL>（**404**，见 [站点归档](../sites/astribot-smoothrl.md)）
- **机构：** 星尘智能（Astribot）
- **作者：** Astribot Team
- **发表 / 上传：** 2026-08（arXiv）
- **平台：** Astribot S1；冻结 π₀.₅ base VLA
- **入库日期：** 2026-09-02

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.29768](https://arxiv.org/abs/2608.29768) | 论文主文 |
| 对照 | [ARLI arXiv:2608.23831](https://arxiv.org/abs/2608.23831) | 异步 VLA + RL（DSRL 噪声舵，非 value-gradient 穿 raw action） |
| 对照 | [RLT 等 §2.3](https://arxiv.org/html/2608.29768) | 同步在线 RL + 冻结 VLA 残差头 |
| 方法 | [action-chunking](../../wiki/methods/action-chunking.md) | chunk 级 MDP 与异步执行 |

## 开源状态（步骤 2.5，2026-09-02 复核）

- **项目页 404；** arXiv **未列** 代码仓库。
- **处理：** wiki 标未开源；`## 源码运行时序图` 标不适用。

## 摘要级要点

- **问题：** VLA/WAM 部署需 **可靠性（在线 RL 微调）** 与 **平滑实时执行（异步 chunk 推理）** 兼得；现有在线 RL 假设同步执行，与部署动力学不匹配。
- **SmoothRL：** value-gradient 范式；每个 action chunk 按帧索引分为 **committed / execution / discarded** 三区；**∇ₐQ 仅经 execution region [n,2n)** 回传；训练 rollout 嵌入与部署相同的异步环。
- **实例化：** 冻结 π₀.₅ + 可训 MLP actor/critic 修正 20 维臂动作；TT-RTC 调度保 latency budget；人类干预以 raw action 残差写入 replay。
- **真机三任务（250 episodes 后）：** 投掷 39%→**94%**；笔帽 8%→**83%**；开箱 30%→**90%**。
- **运动质量：** 投掷 rollout 末端 RMS 加速度 −52%、jerk −47%（相对基线表述，见论文 Fig.1）。

## 核心摘录（面向 wiki 编译）

### 1) 异步环与 latency budget（§3.1）

- 推理 5 Hz、控制 30 Hz → **n=6** 帧；H=32 chunk → committed [0,6)、execution [6,12)、discarded [12,32)。
- 固定 budget：每 chunk 提前 n 步请求；推理提前完成则等待，使 handover 时刻确定。

### 2) 平台（§4.1）

- Astribot S1：25 DoF 移动双臂；三相机 224² + 本体；base 31-dim action（臂/躯干笛卡尔 delta + 夹爪/头）；RL 只改 20 维臂动作。

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
- 交叉补强：[ARLI](../../wiki/entities/paper-arli.md)、[VLA](../../wiki/methods/vla.md)、[action-chunking](../../wiki/methods/action-chunking.md)

## 当前提炼状态

- [x] arXiv HTML §3–4 / Table 1 摘录
- [x] 项目页复核：404，无代码
- [x] 升格 `wiki/entities/paper-smoothrl.md`

# 机器人控制遇到约束冲突，怎样保证动作还能继续？｜附代码与评测入口

> 原始抓取（WebFetch，2026-09-20）

- **来源：** https://mp.weixin.qq.com/s/RozDRLth62xgulo4ccIBMw
- **抓取日期：** 2026-09-20

---

阅读路线

今天更新 11 篇具身智能新论文，覆盖机器人控制与导航、视觉语言与上下文学习、接触与动态稳定操作、人形与多机器人协作等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读 2 篇跟进 8 篇扫读

如果只看一篇，先看 ElastiQP，它直接处理约束控制中“不可行就停摆”的部署痛点；做世界模型导航可接着看 WAVE-Go，研究 VLM 智能体可跟进 GPT-Policy。之后按兴趣扫读 Dreaming the Sound of Contact、WholeBodyWAM、PointZero、RoboVAD、FIERCE、Fetch My Beer、OpenDexGrasp 与 Multi-Humanoid Pickup and Transport。

## 1 约束冲突时，机器人控制器也不能停摆

**ElastiQP: An Always-Feasible QP Solver for Constrained Robot Control**

- 论文：https://arxiv.org/pdf/2609.19080
- 代码：https://github.com/StanfordASL/elastiqp

## 2 导航动作发出后，机器人还要能及时反悔

**WAVE-Go: World-Model Navigation with Adaptive Execution for Wheel-Legged Robots**

- 论文：https://arxiv.org/pdf/2609.18193
- 代码：https://github.com/vigorlee/wave-go

## 3 不改模型参数，也能让机器人从示范中学会做事

**In-Context Robot Learning with VLM Agents (GPT-Policy)**

- 论文：https://arxiv.org/pdf/2609.19138
- 代码：https://github.com/cheng-haha/GPT-Policy
- 项目页：https://cheng-haha.github.io/GPT-Policy

## 04–11 扫读

- Dreaming the Sound of Contact — 2609.19137 — https://dreamingcontactsound.github.io/
- WholeBodyWAM — 2609.18197 — https://zbzyjya.github.io/WholeBodyWAM/
- FIERCE — 2609.18651 — https://github.com/ar-mine/FIERCE
- RoboVAD — 2609.17843 — https://zenodo.org/records/22754659
- PointZero — 2609.19142 — https://github.com/Duisterhof/pointzero
- Fetch My Beer — 2609.18119 — https://fetch-my-beer.github.io/
- OpenDexGrasp — 2609.18117 — https://opendexgrasp.github.io/
- Multi-Humanoid Pickup and Transport — 2609.17824 — http://decmht.github.io/

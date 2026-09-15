# OM-1: Frontier Robot Intelligence, Learned Firsthand from Humans（Reward AI）

> 来源归档（blog / Reward AI 官方）

- **标题：** OM-1: Frontier Robot Intelligence, Learned Firsthand from Humans
- **类型：** blog
- **作者 / 组织：** Reward AI Team / Reward AI
- **原始链接：** <https://www.rewardai.com/blog/OM-1/>
- **发表日期：** 2026-09（博客 Citation 写 Sep 2026）
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch + HTML 交叉核对（含 BibTeX、视频清单）
- **一句话说明：** Reward AI 发布 **Omnibody** 全栈：**Omnibody Hand**（7-DoF 可穿戴采集）+ **One Data Interface**（多模态人类操作数据）+ **OM-1**（仅人类示范、无遥操作/无机载数据的通才操作策略）+ **Control Any Body**（高频 RL 控制层跨工业臂到人形）；宣称 **<30 分钟** 人类数据即可上手新任务。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-09-15） |
|----|-------------------------|
| 公司站 | <https://www.rewardai.com/> — 无 GitHub / Hugging Face / 权重下载入口 |
| OM-1 博客 | 产业发布；正文链向 [DexCap](https://dex-cap.github.io/) 前序工作，**无** OM-1 代码仓 |
| 代码 / 权重 / 数据集 | **确认未开源** |
| 同名混淆 | [OpenMind/OM1](https://github.com/OpenMind/OM1) 为 **另一项目**（机器人 AI runtime / HAL），与 Reward AI OM-1 **无关** |

## 核心摘录（归纳，非全文）

### 主张与定位

- **设计原则：** *One Model, One Data Interface, Any Body* — 采集、学习、控制一体设计，而非模块拼接。
- **人类效率标杆：** 真机操作须达到人的速度、流畅度与效率；仅靠加数据/算力或加速现有系统不够（引用 Anderson「More Is Different」）。
- **前序工作：** 基于团队 [DexCap](https://dex-cap.github.io/)（RSS 2024；Chen Wang 等，含 Li Fei-Fei、C. Karen Liu）的可穿戴灵巧操作 mocap 经验。

### Omnibody Hand（7-DoF 可穿戴）

| 维度 | 要点 |
|------|------|
| 功能导向 | 非人手关节逐点复制；保留 **选接触点、手内重定向、精密/力量抓切换** |
| 自由度 | 7-DoF：拇指-食指捏合 + 拇指/食指屈伸；中指/无名指/小指 MCP 联动 + 拇指屈曲作力量抓 |
| 人体工学 | 适配手型差异；远端屈曲机构吸收指长差，减少逐用户连杆调校 |
| 采集哲学 | 穿戴者按自然方式操作，**不必**为某台机器人 kinematics 扭曲动作 |

### One Data Interface（统一数据接口）

- **被动采集：** 工作、玩耍、烹饪等日常活动均可产可用数据，无需 staged setup 或专人监督。
- **高频传感：** 触觉反馈 + 接近觉（接触前距离）+ 全局快门 in-hand 相机（快速运动下保持视觉上下文）。
- **位姿跟踪：** 视觉-惯性（VI）在快速反转时受视觉更新率限制；叠加 **电磁传感** 得高保真位置信号，算法补偿环境电磁干扰。
- **力测量：** 沿同轨迹记录施力（拉开门、搬重物等），示范含 **路径 + 用力**。
- **定量（博客自报）：** 刚性双 tracker 在八档速度、十次往返实验中，最高速平均 overshoot 误差从 **24.9 mm → 9.5 mm**（约 **60%** 降幅）。

### OM-1（Omnibody Model 1）

| 维度 | 要点 |
|------|------|
| 训练数据 | **仅** 穿戴 Omnibody Hand 的 **无机器人** 人类数据；**无遥操作、无机上经验** |
| 学习路径 | 人类行为 **直接** 生成机器人动作，不经中间机器人 embodiment 路由 |
| 训练阶段 | **无** pre/post 壁垒；首条与最新示范同形，**单阶段** 训同一策略 |
| 跨本体 | 工业机械臂到人形；随人类数据规模与多样性提升 |
| 推理架构 | 面向高效推理的新架构；多模态历史输入仍须 **人类速度** 出动作 |
| 模态 | 图像、触觉、指间接近、手部位姿轨迹；**各模态保持传感器原生采样率**（非统一降频） |
| 时序 | 消费多模态流的时间历史，推理接触/运动/任务进展演化 |
| 动作输出 | 运动方向、速度、力、抓取/移动等关键事件时机；与输入共同构成跨本体 **统一策略接口** |

### Control Any Body（高频控制层）

- RL 在仿真中训练，覆盖速度/加速度相关动力学、外扰、系统延迟。
- **异步时钟：** 控制层高频持续运行，策略推理延迟不中断运动。
- **在线平滑：**  successive 预测间优化过渡，避免高速下的轨迹不连续（投掷、摆动等）。
- 演示：未知阻力的冰箱门拉开、不同重量快递箱搬运等 **接触丰富、动力学不确定** 任务。

### 结论段自报性能

- 全新任务（含挑战动力学与长时域）**<30 分钟** 人类数据即可上手。
- 性能归因于 **采集-传感-学习-推理-控制** 一体化，而非单一策略架构。

## 对 wiki 的映射

- [reward-ai-om1](../../wiki/entities/reward-ai-om1.md) — 本篇升格实体页
- [reward-ai-robotics](../../wiki/entities/reward-ai-robotics.md) — 公司入口页
- [paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti](../../wiki/entities/paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md) — 前序 DexCap 工作
- [hub-cross-embodiment](../../wiki/overview/hub-cross-embodiment.md) — 跨工业臂/人形「Any Body」轴
- [imitation-learning](../../wiki/methods/imitation-learning.md) — 人类示范→策略
- [foundation-policy](../../wiki/concepts/foundation-policy.md) — 闭源通才策略产业对照
- [generalist-gen15-one-shot](../../wiki/entities/generalist-gen15-one-shot.md) — 同赛道闭源 one-shot 对照
- [skild-s1](../../wiki/entities/skild-s1.md) — 视频 ICL 闭源对照

## 可信度与使用边界

- **官方产业博客**，非 peer-reviewed；定量多为自报演示与内部分析。
- **无公开权重/数据/控制栈**；「<30 分钟」「60% overshoot 降幅」等须标为作者立场。
- 与 DexCap 关系：OM-1 为 DexCap 团队后续 **商业全栈** 叙事，DexCap 论文代码与 OM-1 栈 **不等价**。
- 勿与 OpenMind `OM1` runtime 混淆。

## 参考来源

- 原文：<https://www.rewardai.com/blog/OM-1/>
- [Reward AI 公司站归档](../sites/rewardai.md)
- [DexCap 项目页](https://dex-cap.github.io/)

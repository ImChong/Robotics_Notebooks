---
type: entity
project_id: freespeed
tags: [paper, manipulation, action-chunking, speed-control, imitation-learning, control]
status: complete
updated: 2026-10-08
arxiv: "2610.05734"
project: https://yuxuanhu9.github.io/FreeSpeed/
related:
  - ../methods/action-chunking.md
  - ../entities/paper-speedtuning.md
  - ../methods/diffusion-policy.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/freespeed_arxiv_2610_05734.md
  - ../../sources/sites/freespeed_project_page.md
  - ../../sources/repos/freespeed.md
summary: "FreeSpeed 在冻结的生成式策略上推理时重采样动作块，并用方向不一致度限制速度缩放，以保住抓取和放置阶段的成功率。"
---

# FreeSpeed：不重训策略，按动作方向自适应调速

**FreeSpeed** 是训练时不改权重、在推理时调整动作块执行速度的方法：方向变化小的动作段可以更积极地调速，抓取和放置等关键段则保留更多原策略的步长。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 论文评测的视觉-语言-动作策略家族之一 |
| WAM | World Action Model | 论文评测的世界-动作模型家族之一 |
| LIBERO | LIBERO Benchmark | 用于桌面操作仿真的任务套件 |
| I_cos | Cosine Inconsistency | 衡量动作块相邻平移增量方向变化的指标 |

## 为什么重要

模仿学习策略的执行速度来自示范。部署方直接跳过动作或统一拉伸动作时间轴，虽然能改变墙钟速度，却可能让机器人在接触时看到训练中少见的状态。抓取和放置常是轨迹中方向变化集中的阶段，均匀提速会把最需要精细闭环的部分也压缩掉。

FreeSpeed 将速度调节放在策略之后：无需重训基座策略，允许运行中改变速度命令，同时让动作变化大的片段趋近原策略的动作步长。它提供的是执行时间尺度调节，不会补齐错误的抓取意图或替代碰撞、安全控制。

## 核心信息

| 项 | 内容 |
|----|------|
| **作者** | Yuxuan Hu、Shilin Shan、Qiheng Wang、Jinghan Yang、Junqiao Fan、Hao Wan、Jianfei Yang |
| **机构** | 南洋理工大学 MARS Lab；ROKAE Robotics |
| **论文** | arXiv:2610.05734，2026-10-05 提交 |
| **评测策略** | π₀.₅、Fast-WAM、task-specific flow matching |
| **开放状态** | 论文和项目页公开；官方代码仍标注 “Code soon”，没有可验证运行仓库 |

## 方法栈

每次策略重规划先得到一个未来动作块，再根据请求步长 ρ_t 对动作块进行时间重采样：

1. **重采样。** 加速时跳过部分预测动作；减速时把动作拆成方向不变的子动作。
2. **估计方向不一致。** 在动作块中计算相邻平移增量方向的不一致度 I_cos，以相邻对中的最大值代表该动作块。论文用该指标近似当前动作段的任务阶段关键性。
3. **自适应调整步长。** 根据请求倍率与保守倍率做指数插值；当 I_cos=0 时采用请求倍率，方向不一致升高时逐渐回到保守倍率：
   \[
   f_t = f_c + (f_s - f_c) \exp(-\lambda I_{\cos})
   \]
   其中 f_s 是命令对应的缩放因子，f_c 是保守端因子，λ 控制不一致度的衰减速度。
4. **执行缩放结果。** 用 f_t 缩放平移和旋转增量；夹爪指令保持不变。

### 流程总览

\`\`\`mermaid
flowchart LR
  obs["当前观测"] --> policy["冻结策略预测动作块"]
  rate["在线速度命令"] --> resample["时间重采样"]
  policy --> resample
  resample --> score["计算块内方向不一致度"]
  score --> scale["自适应缩放平移与旋转"]
  resample --> scale
  scale --> robot["机器人执行；夹爪命令保留"]
  robot --> obs
\`\`\`

该流程将速度控制放在动作块输出之后：基座策略保持冻结，方向变化只影响后处理幅度。

## 源码运行时序图

**不适用（核查日期：2026-10-08）。** 官方项目页显示 “Code soon”，没有公开源码仓库、README 运行入口或可执行脚本，因此目前无法按真实模块绘制运行时序。论文中的方法流程见上方流程图。

## 工程实践

| 工程项 | 建议 |
|--------|------|
| 先建立 1× 基线 | 对每个任务测原策略成功率和实际执行速率，再比较不同速度命令 |
| 监控关键动作段 | 将块内方向不一致度作为减弱调速的信号，不应当作完整的接触状态估计 |
| 分开记录命令与执行速度 | 请求倍率不等于实际执行倍率；评测需同时报告成功率、命令倍率和测得速率 |
| 保持夹爪语义 | 论文对平移/旋转增量做缩放，夹爪命令不缩放；部署时仍需验证夹爪与控制器时序 |
| 等待官方实现再复现 | 当前仓库未发布；论文中的训练策略、真机配置和评测管线尚不能通过官方代码直接重现 |

**开源结论：** 论文和项目说明可读，官方代码待发布；见 [项目页归档](../../sources/sites/freespeed_project_page.md) 与 [源码状态归档](../../sources/repos/freespeed.md)。

## 实验与评测

论文评估三类策略和 50 个仿真任务。对 π₀.₅、Fast-WAM 与任务专用 flow-matching 策略的不同实验，在各任务保持 1× 基线成功率的设置下，实际执行速率达到 **0.22×–2.53×**。这个范围是按任务筛选保持基线成功的设置，并非任意任务、任意速度命令都保证成功。

真机侧使用 ROKAE Helios 系列人形机器人，以 30 Hz 执行策略动作，测试四项操作任务。六种非参考速度命令上的平均成功率为 **94.0%**，冻结策略 1× 参考为 **93.8%**；逐任务实际执行速率为 **0.38×–1.97×**。这支持了该组实验中速度可调而成功率接近基线的结论，不能外推为所有机器人、控制频率和策略的保证。

## 与其他工作对比

[SpeedTuning](./paper-speedtuning.md) 将执行倍率作为单独策略学习，在其仿真复现中用轻量强化学习选速度；FreeSpeed 不增加速度策略，也不训练新权重，而是从当前预测动作块的方向变化计算后处理尺度。二者都关注“变速时保持任务成功”，但在线信号和训练依赖不同。

与统一对动作块做时间插值相比，FreeSpeed 让高方向不一致片段回到更保守的动作步长。它与 [Action Chunking](../methods/action-chunking.md) 的关系是：动作块仍由基座策略生成，FreeSpeed 只改变块的时间重采样和动作增量执行尺度。

## 结论

**FreeSpeed 的主要工程判断是：机器人能否加速取决于当前动作片段是否允许改变步长，而不应只看全局速度倍率。**

1. **方向变化低时再提速。** 直线、过渡段更适合接近请求速度；抓取和放置等高变化片段应更保守。
2. **不要把 0.22×–2.53× 当作普适能力范围。** 这是论文任务中仍保持逐任务 1× 成功率的实际速率区间。
3. **真机结果支持这组设置。** 四项任务平均成功率与冻结基线接近，跨任务及硬件泛化仍需单独验证。
4. **它是轻量执行后处理。** 不改策略权重、不修复感知或抓取语义，也不替代低层安全约束。
5. **复现受代码发布状态限制。** 项目页核查时仍显示 “Code soon”，实验数字目前应视为论文报告。

## 局限与风险

- 方向不一致只是任务阶段关键性的代理信号；同一几何方向变化也可能来自避障、噪声或控制误差。
- 项目页指出加速上限受任务中自由运动段占比限制；接触密集任务未必能得到明显加速。
- 公式需要配置保守端因子与衰减系数，论文报告的表现不能直接替代目标机器人上的参数验证。
- 官方代码、数据入口及许可证尚未发布，当前不具备一键复现实验的条件。
- 速度后处理只改变已有动作块的执行尺度，不能纠正目标识别、抓取点或策略输出本身的错误。

## 关联页面

- [Action Chunking](../methods/action-chunking.md) — 基座策略输出的动作块接口与滚动执行
- [SpeedTuning](./paper-speedtuning.md) — 另一路线：学习独立速度策略
- [Diffusion Policy](../methods/diffusion-policy.md) — 连续生成式动作策略背景
- [Manipulation](../tasks/manipulation.md) — 抓取、放置等操作任务背景

## 参考来源

- [FreeSpeed 论文归档](../../sources/papers/freespeed_arxiv_2610_05734.md)
- [FreeSpeed 项目页归档](../../sources/sites/freespeed_project_page.md)
- [FreeSpeed 官方源码状态](../../sources/repos/freespeed.md)
- [arXiv:2610.05734](https://arxiv.org/abs/2610.05734)
- [FreeSpeed 项目页](https://yuxuanhu9.github.io/FreeSpeed/)

## 推荐继续阅读

- [FreeSpeed 项目页：方法与可视化实验](https://yuxuanhu9.github.io/FreeSpeed/)
- [Action Chunking](../methods/action-chunking.md) — 理解动作块预测与执行层的差别
- [SpeedTuning](./paper-speedtuning.md) — 对照学习型速度控制策略
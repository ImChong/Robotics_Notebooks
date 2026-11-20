# Robot Parkour Learning（Robot Parkour Learning，HMI P130）

> 来源归档（ingest；最近复核：2026-11-20）— 策展解读编译，非原文镜像；下方「核对结果」依据原文全文

- **通读：** 已完成
- **核对范围：** 2026-11-20 通读 `arXiv:2309.05665v2` 全文（正文 §1–§5、参考文献、附录 A–E 含 Table 3–11 奖励 / 随机化 / PPO / 网络参数与 Figure 9），核实 Table 1 障碍范围、Table 2 仿真成功率（Ours 86/80/100/73/100）、实机 A1 攀 0.40 m 80% / 跨 0.60 m 80% / 钻 0.2 m 90% / 窄缝 0.28 m、`r_skill`/`r_penetrate` 公式与 ±0.05 课程、DAgger BCE 目标、48×64 深度 10 Hz / 策略 50 Hz / PD 50,1 / 25 Nm 限幅；并读官方仓库 README、`legged_gym/README.md` 与 `scripts/{train,collect,play}.py` 入口。

- **标题：** Robot Parkour Learning
- **短名：** Robot Parkour Learning
- **类型：** paper / hmi-papers / Locomotion与运动先验
- **HMI ID：** P130
- **年份：** 2023
- **原文：** https://arxiv.org/abs/2309.05665
- **代码：** https://github.com/ZiwenZhuang/parkour
- **项目页：** https://robot-parkour.github.io/
- **入库日期：** 2026-07-31
- **一句话说明：** 用直接配点启发的软→硬动力学约束课程先让策略发现可行动作，再蒸馏成接收深度的单一视觉四足跑酷策略。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P130](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P130.md)

## 开源状态（步骤 2.5）

- **结论：** 已开源（与 Extreme Parkour 同仓生态，但论文问题设定不同）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

Robot Parkour解决的是一个探索难题：攀高台、跨大沟、钻低洞和侧身穿缝都需要短时间内做出极端动作，但如果障碍从训练开始就是不可穿透的，随机策略几乎拿不到前进奖励；为每项技能手写一套动作模板又很难扩展。作者用“先让障碍可穿透，再恢复真实碰撞”的课程替代参考动作。

**对 wiki 的映射：** [`wiki/entities/paper-robot-parkour-learning.md`](../../wiki/entities/paper-robot-parkour-learning.md)

### 摘录 2

系统分别训练攀爬、跨沟、低姿穿越、侧身挤过和奔跑专家。第一阶段障碍允许穿透，碰撞点进入障碍的深度形成连续惩罚，策略即使还不会完整越障，也能从向前运动与较小穿透中获得梯度。课程逐渐提高障碍和穿透约束难度。第二阶段再换成不可穿透的硬碰撞环境细调，让动作满足真实接触动力学。整个过程只使用简单的前进、能耗和存活类奖励，没有动物参考动作，也没有AMP判别器。

**对 wiki 的映射：** [`wiki/entities/paper-robot-parkour-learning.md`](../../wiki/entities/paper-robot-parkour-learning.md)

### 摘录 3

专家训练时可以看到特权地形和物理状态，最后通过DAgger蒸馏成一个循环视觉策略。部署策略输入机载深度与本体历史，自动判断应当爬、跳、钻还是侧身，不需要外部技能切换器。作者还专门模拟深度缺失、噪声和延迟，并让视觉预处理尽量匹配真机相机。

**对 wiki 的映射：** [`wiki/entities/paper-robot-parkour-learning.md`](../../wiki/entities/paper-robot-parkour-learning.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-robot-parkour-learning.md`](../../wiki/entities/paper-robot-parkour-learning.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)

## 核对结果（依据原文 arXiv:2309.05665v2）

- **方法：** 五个专项技能（climb / leap / crawl / tilt / run）各自 PPO 训练：先软动力学（障碍可穿透，`r_skill + r_penetrate`，穿透奖励乘 `v_x`，自动课程按穿透奖励阈值以 0.05 步长调节难度分数 `s`），再硬动力学微调（仅 `r_skill`）；run 跳过软阶段。之后 DAgger 蒸馏为单一 CNN+GRU+MLP 视觉策略，散度为 BCE。
- **仿真（Table 2）：** Ours 成功率 86/80/100/73/100，Oracle 95/82/100/100/100，Blind 0/0/13/0/100，MLP 0/1/63/43/100，No Distill 0/0/73/0/100，Oracles w/o Soft Dyn 0/0/93/86/100。
- **实机：** A1 定量（每项 10 次）；Go1 仅说明同管线可用。户外跨 0.6 m、连续 0.42 m 台阶。
- **局限（原文 §5）：** 仿真环境需手工构建；未来用 3D 视觉自动生成环境、尝试 RGB 输入。
- **开源：** 官方仓库 ZiwenZhuang/parkour 含 legged_gym + rsl_rl 训练 / 蒸馏与 Go1 / Go2 部署；README 注明 Go1/Go2 配置不保证复现论文性能，A1 部署代码待办。

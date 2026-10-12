# Humanoid Horizon 论文来源归档

## 元数据

- **题目：** Humanoid Horizon: Extending Task Horizon in Whole-Body Loco-Manipulation via Parallel Training, Dynamic Starting, and Reward Gating
- **作者：** Haozhuo Zhang, Qiang Zhang, Jian Tang, Mingzhe Ni, Michele Caprio, Angelo Cangelosi（通讯）, Wei Pan（通讯）
- **单位：** 曼彻斯特大学；中国科学技术大学；X-Humanoid；华威大学；纽卡斯尔大学
- **预印本：** https://arxiv.org/abs/2610.08320
- **HTML：** https://arxiv.org/html/2610.08320v1
- **项目页：** https://haozhuo-zhang.github.io/Humanoid-Horizon-project-page/（见 [sources/sites/humanoid_horizon_project.md](../sites/humanoid_horizon_project.md)）
- **版本：** v1，2026-10-06，cs.RO / cs.GR
- **代码：** 截至 2026-10-12 项目页与论文均未列出代码仓库，也无「将发布」声明。

## 论文要解决的问题

多物体连续搬运（导航、抓取、搬运、放置，且每个物体之间不重置）中，单策略强化学习有两种失败：易奖励偏置（早期阶段被过度优化）与灾难性遗忘（后期阶段训练损害已学好的前期行为/已放置物体）。

## 方法与训练信息

1. **并行训练：** 单物体预训练后，16,384 个环境平均分给 S 个阶段流，共享同一策略参数；第 1 流从原始初始状态起，第 i>1 流从缓冲 B_i 起（初始为机器人位于前一物体目标附近的默认姿态代理状态）。
2. **动态起始：** 每个 epoch 末，阶段 i 的终止状态覆盖 B_{i+1}；下一 epoch 的阶段 i+1 从该缓冲初始化。
3. **奖励门控：** 对阶段 i>1，一旦前一物体 o_{i-1} 离目标超过容差 ε，其后全部奖励置零（ε 数值在可见正文中未给）。
4. **任务奖励：** 八项塑形项权重和为 1（接近速度/位置各 0.1、抓取接近 0.1、抬升 0.1、物体搬运速度 0.2、路点距离 0.2、目标接近 0.1、目标朝向 0.1），外加 AMP 判别器奖励权重 1.0。
5. **Teacher：** 特权状态 250 维 + 200 点基点集（MLP 600→512→512→128）；actor/critic MLP 1024-1024-512；AMP 判别器取 10 帧×131 维。PPO：4 epoch/iter、horizon 32、8 mini-batch、γ 0.98、λ 0.95、clip 0.2、Adam lr 5e-5、最多 16,000 epoch；episode 300 步（单物体）/ 600 步（双物体）。
6. **VLA Student（DAgger）：** 本体感知 223 维 + 256×256 egocentric RGB 或深度（90° 水平视场）+ 冻结 BERT-base 语言嵌入（768 维）+ 上一步动作，EfficientNet-B0 图像骨干，残差 MLP 输出 28 维；DAgger β 从 0.998 指数衰减，558 环境、回放 100,000、batch 600、lr 5e-4、最多 30,000 epoch，MSE 对齐 Teacher 动作。
7. **平台：** 15 刚体 28 DoF 人形，PD 位置目标，30 Hz 控制 / 60 Hz 仿真，NVIDIA IsaacGym（PhysX），2×A100 80GB。

## 论文报告的结果

- **LHM-Humanoid 基准：** 350 个训练/评测场景、66 个未见场景、79 类物体（25 类可作放置目标），卧室/客厅/厨房/仓库；每个 episode 两次取-搬-放，物体根位置距目标 0.2 m 内为成功。
- **350 训练场景（Succ1/Succ2/SuccAll）：** Humanoid Horizon 96.6/81.3/80.4；LHM-Humanoid 87.4/72.6/71.1；Curriculum RL 88.3/48.5/47.2；TokenHSI 40.5/31.5/27.6；HumanVLA 42.3/36.7/29.9；Hierarchical RL 37.5/25.0/20.8；InterMimic 20.6/9.8/7.5；端到端 RL 2.3/0.0/0.0。
- **66 未见场景 SuccAll：** 本方法 78.0；LHM-Humanoid 61.6；Curriculum RL 39.5。
- **消融：** 去动态起始 67.7 / 去奖励门控 69.6（训练场景 SuccAll），对完整的 80.4。
- **VLA 蒸馏（SuccAll）：** HH(RGB) 75.6、HH(Depth) 73.7；LHM-Humanoid 63.7；去动态起始 51.4；去门控 58.6。
- **多于两个物体（SuccAll，S1…S5）：** 本方法 60.2（96.5/85.2/82.2/75.4/66.5）；LHM-Humanoid 18.1（五教师版 25.8）；去动态起始 49.8；去门控 38.9；Curriculum RL 3.2。
- **种子方差（附录 B，训练期指标）：** 三种子 16,000 epoch，训练 SuccAll 87.6±0.3%，Succ1 92.5±0.3%；训练阈值 0.1 m 比评测 0.2 m 严格，不可直接比较。

## 复现与解释边界

- 无公开代码；项目页「Videos」栏为空，仅有模板、示意图。
- 基线由作者改造（HumanVLA 拆为 700 子任务，Hierarchical RL 使用特权 FSM），对比依赖这些适配。
- 仅仿真；VLA 研究为探索性，真机部署列为未来工作。失败模式：Succ1 到 Succ2 差距、冻结动作式抓取失败。
- 本次读取到正文约前 100,000 字符，附录 C（评测种子方差）与 D 未读。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-humanoid-horizon-long-horizon-loco-manipulation.md
- 任务入口：wiki/tasks/loco-manipulation.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.08320>
- arXiv HTML 全文：<https://arxiv.org/html/2610.08320v1>
- 项目页：<https://haozhuo-zhang.github.io/Humanoid-Horizon-project-page/>

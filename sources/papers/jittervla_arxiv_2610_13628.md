# JitterVLA 论文来源归档

## 元数据

- **题目：** JitterVLA: A Replay Benchmark for Motion Quality in VLA-Driven Human-Robot Manipulation
- **作者：** Faith Ogunwoye, Didar Seyidov, Konstantin Gubernatorov, Dzmitry Tsetserukou
- **单位：** Intelligent Space Robotics Laboratory, Skolkovo Institute of Science and Technology（Skoltech），Moscow
- **预印本：** https://arxiv.org/abs/2610.13628
- **HTML：** https://arxiv.org/html/2610.13628v1
- **PDF：** https://arxiv.org/pdf/2610.13628
- **版本：** v1，2026-10-09，cs.RO
- **项目页：** https://faithogunwoye.github.io/JitterVLA/（归档：[jittervla-github-io.md](../sites/jittervla-github-io.md)）
- **数据：** https://huggingface.co/datasets/armteam/JitterVLA-teleop-episodes
- **代码：** 论文称审稿期后在 Apache-2.0 下发布，截至 2026-10-12 无公开仓库
- **入库日期：** 2026-10-12

## 论文要解决的问题

VLA 策略的稀疏动作目标到达底层控制器时间晚且不规则，即使预测合理，也会在人旁边产生突兀运动。JitterVLA 固定命令序列与到达时序，只更换控制器，把“控制器本身”的影响隔离出来。

## 方法与协议

- **成对回放：** 命令源与延迟实现固定，仅改变控制器；统计按源单位（episode / 人 / 种子）配对，95% 区间用 5,000 次单位自助法。
- **时序管线：** 来自一张 RTX 5090 的实测 VLA 推理耗时进入单工作者 FIFO 队列；网络延迟三选一：零延迟、Gamma 延迟（均值 0–400 ms，形状参数 2）、12 小时实测网络轨迹（LAN、互联网往返、冷启动 HTTPS）；最新到达的目标生效。
- **被控对象：** 通用三轴质点（3 kg，临界阻尼阻抗，20 ms 力滞后，3000 N/s 力速率上限，1 ms 步长），明确不是 UR3 模型。
- **控制器：** B0 阶梯保持；B2 在一个命令周期上线性插值；A2 = B2 + 熵门控刚度；B3 Ruckig 限加加速度轨迹生成（25 m/s³）；另有 8 档固定刚度的常增益网格作为“可达最优”基线。
- **指标：** p99 与 RMS jerk、相对无延迟参考的跟踪 RMSE，另用 LDLJ 与 SPARC 作平滑性检查。
- **数据源：** 10 条合成到达、20 人手部轨迹（3D-ARM-Gaze，5 Hz 目标）、OpenVLA / BridgeData V2（80 + 200 episode，逐 token 熵）、pi0-FAST / DROID（80 + 199，10 步 chunk 熵）、pi0.5 / DROID（80 + 199，15 步流匹配 chunk，无熵）；共 838 个 VLA episode，其中 598 个为新留出抽样；pi0.5 在 v1.5 看过结果后加入；总计 69,988 次成对回放。UR3（CB3，125 Hz）为 30 + 30 次回放遥操作轨迹。

## 论文报告的结果

- **插值（B2 对 B0，零额外延迟）：** Δjerk（m/s³）/ ΔRMSE（cm）：合成 −504 / +2.39；人手 −521 / +2.32；OpenVLA −202 / +0.78（新抽样 −193 / +0.77）；pi0-FAST −371 / +0.28（−343 / +0.26）；pi0.5 −374 / +0.56（−363 / +0.54）。
- **人手 + 200 ms 平均延迟：** B0 为 615 m/s³ / 9.91 cm；B2 为 106 m/s³ / 11.87 cm（−83%）；B3 为 23.5 m/s³ / 14.79 cm；A2 为 150 m/s³ / 16.72 cm，两轴均不如 B2。
- **延迟：** 固定 0 / 200 / 400 ms 时阶梯控制 p99 jerk 约 535 / 534 / 533；200 ms 延迟的抖动将其抬到 623。实测轨迹上 pi0-FAST 的 B2−B0 为 −368（LAN）、−350（互联网）、−331（HTTPS）m/s³。LDLJ 与 SPARC 在所有测试格中都偏向 B2。
- **熵相关：** OpenVLA 的 token 熵与目标步长呈弱负相关 ρ=−0.065（新抽样 −0.089）；pi0-FAST 的 chunk 熵与 chunk 边界跳变呈正相关 ρ=0.221（新抽样 0.217）。A2 对 B2：pi0-FAST 约低 208 m/s³ p99 jerk，跟踪多 0.18 cm；OpenVLA 无收益（报告对比 +2.3 [−0.7, 5.2]，新抽样 +3.4 [1.9, 4.8]，单位文中未说明）。两条管线上熵的效应符号相反。
- **复现：** pi0-FAST 的 28 个预注册控制器对比全部复现；OpenVLA 的 28 个中 22 个复现，未复现的多接近零。
- **UR3：** 冻结的 v1.2 协议下位置派生的 p99 滤波 jerk 为 77.7–81.9 m/s³，计划对比未分辨；探索性 v1.2a 加装工具法兰加速度计，直接模式下三次配对重复中 B0 均高于 B2，平均差 3.07 m/s³（无额外延迟）和 5.79 m/s³（200 ms 平均延迟），B2 降低 3.0–14.0%；plant 模式差异符号混杂。加速度计静止时 p99 为 3.79，运动时 43.05–51.52，增益与带宽未标定。

## 复现与解释边界

- 作者自述局限：单一通用质点对象；VLA 预测为开环、无任务成功率；三条管线在模型、数据集、表示、熵定义、不连续事件上都不同，不构成受控比较；时序来自单 GPU 和单个实验室网络；UR3 样本小、加速度计未标定，v1.2a 是在 v1.2 零结果之后设计的；指标只是运动质量代理，未测舒适度、信任或感知安全。
- 因此不能把插值带来的 jerk 下降直接当作人机交互安全证明。
- 数据发布状态：HF 数据集页面可见，但抽取到的页面显示默认子集 12 行、总大小 2.67 GB、无数据集卡片，项目页写 29 个 episode；两者不一致，未核实原因。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-jittervla-replay-benchmark.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md

## 一手来源

- <https://arxiv.org/abs/2610.13628>、<https://arxiv.org/html/2610.13628v1>
- 项目页：<https://faithogunwoye.github.io/JitterVLA/>

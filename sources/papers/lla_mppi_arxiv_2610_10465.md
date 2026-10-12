# LLA-MPPI 论文来源归档

## 元数据

- **题目：** LLA-MPPI: Rapidly Adaptive Whole-body Control of Legged Robots with GPU-Accelerated Parallel Simulations
- **方法：** Look-Back, Look-Ahead MPPI（LLA-MPPI）
- **作者：** Sebin Jung, Maitham F. AL-Sunni, Juan Alvarez-Padilla, Zachary Manchester, Changliu Liu, John M. Dolan（前两位共同一作）
- **单位：** arXiv 页面未列出机构（周报写作 CMU / MIT，arXiv 页面未核实）
- **预印本：** https://arxiv.org/abs/2610.10465
- **HTML：** https://arxiv.org/html/2610.10465v1
- **版本：** v1，2026-10-07，cs.RO / eess.SY
- **项目页：** https://lla-control.github.io（2026-10-12 核查：仅标题 “Look Back, Look Ahead” 与 “coming soon”）
- **代码：** 截至 2026-10-12 未发布；arXiv 页面无仓库链接

## 论文要解决的问题

实时腿式全身控制器通常围绕一个固定标称模型规划；载荷、摩擦、坡度或某条腿失效使接触动力学改变后，性能迅速退化。离线领域随机化 RL 需要预先覆盖变化范围，经典参数估计则依赖连续参数的显式辨识。论文把「适应」改写成：在一组 GPU 批量接触仿真器里挑选最能解释最近运动的那个，再让 MPPI 通过它规划。

## 方法与训练信息

- **仿真器库（look-back）：** K 个带不同质量、惯量、摩擦、执行器限幅或结构条件（哪条腿受损）的完整接触仿真器，放入同一个批量 MuJoCo Warp 模型（MuJoCo 3.4.0 + MuJoCo Warp 1.12.0，不是 MJX），整步捕获为 CUDA graph，推进整个库只需一次 graph launch。连续参数网格化、离散结构假设枚举，取笛卡尔积。
- **选择规则：** 每个候选预测最近一次实测转移，残差为预测误差的加权范数，在长度 W 的滑窗上取平均；最低者成为挑战者，仅当其得分低于 γ × 现任者得分时才替换，可附加持续步数条件；窗口未填满前使用标称模型。
- **规划（look-ahead，CPU）：** 全身 MPPI，N 条采样序列、预测时域 H，对 J 个样条控制点加高斯噪声并裁剪到执行器范围；代价含跟踪项、PD 力矩指令项、任务项与终端系数 ρ，指数加权（温度 λ）更新标称序列，施加首个指令后平移序列热启动。
- **异步解耦：** look-back 异步运行，规划所用的选择可能滞后最新转移至多一个 look-back 周期。
- **无离线训练。** N、H、控制频率在可见正文中未给出。

## 论文报告的结果

四个仿真任务，每任务每控制器 50 次随机试验：

1. 非对称载荷（K=10,000，两侧各 0–4 kg 网格，W=20，γ=0.999）；
2. 腿锁死且无故障信号（K=5，W=10，γ=0.5，三步持续）；
3. 斜坡+摩擦衰减（K=10,000，坡度 0–20°、摩擦 0.1–1.0，W=20，γ=0.99）；
4. 变质量推箱（K=100，1–15 kg，W=20，γ=0.999）。

Table I（每方法 200 回合）：

| 方法 | 成功率（95% CI） | ≥90% 任务数 | 跌倒率 | 求解时间 |
|---|---|---|---|---|
| Nominal MPPI | 25.5% [20.0, 32.0] | 1/4 | 65.3% | 16.57 ms |
| Oracle-MPPI | 98.5% [95.7, 99.5] | 4/4 | 0.0% | 17.77 ms |
| LLA-MPPI | 97.5% [94.3, 98.9] | 4/4 | 0.0% | 17.60 ms |
| DOB-MPPI | 42.5% [35.9, 49.4] | 1/4 | 33.3% | 15.43 ms |
| CPE（粒子滤波） | 74.0% [67.5, 79.6] | 1/4 | 21.3% | 18.54 ms |

（跌倒率不含腿锁死任务。）

Unitree Go2 真机（OptiTrack 提供基座位姿/速度与箱子位置，关节来自编码器，无离线参数标定，未报告试验次数）：行走中加 2.3 kg 右侧载荷、后左腿中途失能、推箱中质量两次增加；变质量推箱 LLA-MPPI 约 74 s 完成，标称 MPPI 约 166 s。

与 RL 的对比（Table II，非对称载荷）：PPO-DR（nominal）离线训练 48.5 min，达 90/95/99% 需 46M/46M/69M 样本；PPO-DR（heavy）54.7 min，92M/115M/161M；LLA-MPPI 无离线训练，在线每步 14.4 ± 1.2 ms（RTX 4070 Laptop GPU）。RL 对比的量化成功率未报告。

消融：窗口 W 与成功/误差呈 U 形（短窗口噪声敏感、长窗口数据陈旧）；库越大误差越小，测试最佳 (K, W) = (257, 25)；K 增大时 look-back 耗时在 GPU 饱和前几乎持平。

## 复现与解释边界

- 假设库有限且在设计时固定；近期运动没有激励相关不确定性时候选不可区分（例如静止且不接触物体的质量）。
- 选择可能滞后一个 look-back 周期。
- 作者列出的后续工作：自适应库细化、对多候选的不确定性感知规划、更多接触条件下的真机验证。
- 真机无试验次数；N/H/控制频率未在可见正文给出，复现需等待官方代码。
- 项目页仅为占位，代码与视频未发布。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-lla-mppi-legged-adaptive-wbc.md
- 方法入口：wiki/methods/mppi.md
- 任务入口：wiki/tasks/locomotion.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10465>
- arXiv HTML 全文：<https://arxiv.org/html/2610.10465v1>
- 项目页：<https://lla-control.github.io>

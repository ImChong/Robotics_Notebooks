# VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception（arXiv:2502.14814）

> 来源归档（ingest）

- **标题：** VB-Com: Learning Vision-Blind Composite Humanoid Locomotion Against Deficient Perception
- **类型：** paper / humanoid / locomotion / perceptive-locomotion / policy-composition
- **arXiv abs：** <https://arxiv.org/abs/2502.14814>
- **PDF：** <https://arxiv.org/pdf/2502.14814>
- **版本：** v1 2025-02-20；v2 2025-06-01（本摘录对齐 v2）
- **会议：** ICRA 2026
- **项目页：** <https://renjunli99.github.io/vbcom.github.io/> — 归档见 [`sources/sites/vbcom-github-io.md`](../sites/vbcom-github-io.md)
- **视频：** <https://youtu.be/f9iUE3v7I-8>
- **代码：** 项目页按钮为 **Code (coming soon)**，`href=""`；截至入库日 **无官方可运行仓库**
- **机构：** 上海人工智能实验室（Shanghai AI Lab）；香港大学（HKU）；上海交通大学（SJTU）；浙江大学（ZJU）；香港中文大学（CUHK）
- **作者：** Junli Ren、Tao Huang、Huayi Wang、Zirui Wang、Qingwei Ben、Junfeng Long、Yanchao Yang、Jiangmiao Pang†、Ping Luo†
- **平台：** Unitree G1、Unitree H1（仿真 + 真机）
- **分类（Paper Notebooks）：** 05_Locomotion
- **入库日期：** 2026-08-28
- **最后更新：** 2026-08-28
- **一句话说明：** 用 **仅本体可部署的回报估计器** 在 **视觉策略 πv** 与 **盲策略 πb** 之间切换，让人形在高程图失效、动态障碍或漏踩缺口时切到盲走恢复，而不是把噪声硬塞进单条感知策略。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://renjunli99.github.io/vbcom.github.io/> | 真机避障/跨栏/缺口恢复视频；Code coming soon |
| 视频 | <https://youtu.be/f9iUE3v7I-8> | 项目页嵌入的演示片 |
| 高程图实现来源 | PIM（Long et al., arXiv:2411.14386） | 论文称机载 elevation map 硬件实现详见 PIM |
| Paper Notebooks 进度锚点 | [`humanoid_pnb_vb-com-learning-vision-blind-composite-humanoid.md`](./humanoid_pnb_vb-com-learning-vision-blind-composite-humanoid.md) | 姊妹仓库 progress 条目溯源 |

## 摘要级要点

- **问题：** Blind policy 只靠本体、鲁棒但慢，往往要先碰撞再改步；Vision policy 能前瞻，但真机噪声、传感器失效、仿真缺动态/可变形地形会 **误导** 高自由度、易摔的人形。
- **方法：** **VB-Com** = 同奖励/动作空间下分别训练的 πv（本体 + 机器人中心高程图）与 πb（仅本体），再用两个 **仅吃历史本体** 的回报估计器 πe_v / πe_b 做 Q-informed 合成。
- **切换：** 默认走 πv；当 `Ĝe_v > Ĝe_b > Gth` 不成立时切到 πb。`Gth` 为近 5 步盲策略回报均值减阈值 α；视觉回报再做 5 步平滑；高关节速度时禁止切换。
- **训练：** 单阶段 PPO；goal-reaching 而非速度跟踪；非对称 critic（特权线速度 + 更大高程图）；在线回归下一时刻速度 vt+1 的轻量状态估计器。
- **评测噪声：** Gaussian / 前向或侧向 shifting / 竖直 floating；课程到 100% 噪声。
- **仿真（Table II，最大课程地形）：** 0% 噪声 VB-Com 目标完成 **84.05%** vs Vision **73.57%**；100% 噪声 VB-Com **84.81%** vs Vision **48.71%**、Blind **83.76%**、Noisy Perceptive **80.52%**。Blind 碰撞更多；Noisy Perceptive 在栏上较好、缺口上不如 VB-Com。
- **真机：** G1 / H1；静态障碍走视觉前瞻，高速迎面行人高程图来不及 → 碰撞后切盲策略躲开；零高程输入下连续栏与漏踩缺口恢复。
- **开源：** **宣称将开源 / 截至 2026-08-28 项目页未列 GitHub**（Code coming soon）。

## 核心摘录（面向 wiki 编译）

### 1) 观测与命令

- 命令（沿 Extreme Parkour 航点）：`ct = [d1, d2, vc]`，指向接下来两个目标方向 + 线速度。
- 本体 op：关节角/速度、基座角速度、机体系重力方向。
- 视觉 ov：机器人中心高程图；actor **1.2 m × 0.7 m**，critic **1.6 m × 1.0 m**（加速课程爬升）。
- 特权 critic：准确线速度 vt；critic 侧本体/高程不加噪声。
- πb 训练时不给 ov。

### 2) 回报估计与合成

- 共享 (S, A, R) 的策略集上，按 Q 值构造 categorical：`Pw(i) ∝ exp(Qi/α)`。
- 估计器输入 **仅历史本体** `op,t−H:t`（可上真机、避免被失效高程图再误导）。
- 引入切换周期 T，用 λ-return 估 `st:t+T` 的加权回报；监督信号来自 GAE：`Gπ = Â + V`。
- 部署规则（论文式 7–8）：视觉回报高于盲回报且二者都高于 `Gth` 才执行 πv，否则 πb。`Gth = mean(近 5 步 Ĝe_b) − α`。
- 消融：无 `Gth` 目标完成掉到 **48.48%**；α=0.5 约 **85.76%**。T 过短（1/5 step）两策略互相打断轨迹；T=50 优于 T=1/5/100 与 MC 回报回归。

### 3) 机器人与碰撞

- G1：**20** 维全身动作（上 10 + 下 10）；H1：**19** 维（上 8 + 下 10 + 躯干 1）。
- 只开子集碰撞链 `Cel`：G1 开手碰撞，让盲策略在感知失效时伸手探障。
- 头戴 LiDAR → elevation map；积分窗口使动态场景成为「感知缺失」主场景。
- πv 训练噪声：10% Gaussian + 过去 0.5 s 内随机感知延迟（评测噪声可到 100%，远超训练）。

### 4) 地形尺度（Table I，最大课程）

| 地形 | Length (m) | Width (m) | Heights (m) |
|------|------------|-----------|-------------|
| Gaps | (0.6, 1.2) | **(0.6, 0.8)** | (−1.8, −1.5) |
| Hurdles | (0.8, 1.0) | (0.1, 0.2) | **(0.2, 0.4)** |
| Obstacles | (0.2, 0.4) | (0.2, 0.4) | (1.4, 1.8) |

粗体为论文标出的难度主因子。每回合 8 个目标点；每方法 10×3 次。

### 5) 开源核查（步骤 2.5，截至 2026-08-28）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线；Paper / arXiv / Video 有链 |
| GitHub / 训练代码 | **未列**；按钮文案 *Code (coming soon)*，`href` 为空 |
| 权重 / 数据集 | 未见 |
| 复现入口 | **不适用**（无可运行官方实现） |

## 对 wiki 的映射

- 升格实体：[paper-notebook-vb-com-learning-vision-blind-composite-humanoid](../../wiki/entities/paper-notebook-vb-com-learning-vision-blind-composite-humanoid.md)
- 挂接枢纽：[楼梯与障碍 Locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)
- 观测分类：[人形运控策略的观测输入](../../wiki/concepts/humanoid-policy-observation-inputs.md)
- 任务：[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)
- 感知对照：[RPL](../../wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)
- 同作者高程图：[PIM 占位页](../../wiki/entities/paper-notebook-learning-humanoid-locomotion-with-perceptive-int.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] 项目页源码核查（coming soon）
- [x] wiki 页面映射确认

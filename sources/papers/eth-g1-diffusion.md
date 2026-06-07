# Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking（arXiv:2604.17335）

> 来源归档（ingest · ETH RSL + SFU · 全身感知 locomotion）

- **标题：** Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking
- **类型：** paper / humanoid / diffusion / motion-tracking / RL / perceptive-locomotion / whole-body
- **arXiv abs：** <https://arxiv.org/abs/2604.17335>
- **arXiv HTML：** <https://arxiv.org/html/2604.17335v1>
- **PDF：** <https://arxiv.org/pdf/2604.17335>
- **项目页：** <https://wholebodylocomotion.github.io/>
- **作者：** Zewei Zhang, Kehan Wen, Michael Xu, Junzhe He, Chenhao Li, Takahiro Miki, Clemens Schwarke, Chong Zhang, Xue Bin Peng, Marco Hutter
- **机构：** ETH Zurich Robotic Systems Lab；Simon Fraser University；ETH AI Center；EPFL
- **硬件平台：** Unitree G1（机载 Livox MID360 LiDAR + IMU、Jetson Thor + Orin）
- **入库日期：** 2026-06-07
- **一句话说明：** 三阶段管线——人体运动重定向与增广 → 离线预训练扩散运动生成器 + DeepMimic 式 RL 全身跟踪器 → 冻结生成器的闭环 RL 微调；真机 onboard 感知与计算下完成箱攀、跨栏、楼梯与混合地形穿越。

## 摘要级要点

- **痛点：** 纯 reward shaping RL 易收敛为下肢主导、缺乏全身协调；纯 motion tracking 只能回放固定参考，难以按感知在线改写运动以适配地形。
- **对立面：** 多专家蒸馏 / 技能组合需精心设计数据分布与切换；开环运动扩散直接上真机易出现 foot sliding、时间不连续等运动伪影。
- **本文定位：** **扩散生成参考 + RL 物理跟踪** 的分层框架，生成器按目标朝向与地形高程图在线产出参考，跟踪器负责物理可行执行并在闭环微调中吸收生成误差。
- **数据：** 初始约 5 分钟人体/视频运动（攀 50 cm 箱、跨 35 cm 栏、50 cm 跳下、20 cm 楼梯、全向行走），经 GVHMR 重建 + 接触约束 IK 重定向 + DeepMimic 式跟踪策略精炼；运动学增广（障碍尺度、路径随机小箱）扩展至约 **1 小时** 离线数据。
- **预训练：**
  - **Tracker：** IsaacLab + PPO，模仿奖励 + 正则；观测含参考状态、5 帧本体感受、地形高程扫描；动作 23 维目标关节位置。
  - **Generator：** MDM 架构扩散模型，0.5 s 视界（25 帧），条件为朝向向量、地形扫描、过去 2 帧运动特征；含速度/关节一致/地形穿透等几何损失。
- **闭环微调：** 生成器**冻结**，以机器人过去 2 帧状态（非自回归）条件化；跟踪器在更随机朝向与多样地形（楼梯 15–25 cm、连续栏 25–55 cm、箱/金字塔 30–85 cm）上继续 PPO，奖励加 **heading tracking**；推理仅 **2 步去噪**；训练/部署期对生成与观测注入噪声。
- **真机：** DLIO 位姿 + Elevation Mapping CuPy 地形；被动颈部补偿；TensorRT 加速生成器至约 **20 ms**；0.25 s receding-horizon 更新参考（训练 0.5 s 视界）。

## 核心摘录（面向 wiki 编译）

### 三阶段训练（Fig. 2）

| 阶段 | 内容 | 关键设计 |
|------|------|----------|
| (a) 数据采集 | 视频/数据集 → GVHMR → IK 重定向 → 对齐地形 → tracker 精炼轨迹 | 不直接用原始重定向/优化轨迹，而用跟踪策略录制的物理可行轨迹 |
| (b) 预训练 | 扩散生成器 + 全身跟踪器并行在离线数据上训练 | 跟踪器已含外感受地形输入，为微调铺路 |
| (c) RL 微调 | 冻结生成器闭环运行 + 跟踪器 PPO 微调 | 生成器作「技能组合模块」，跟踪器作「运动滤波器」抑制不安全执行 |

### 与相邻路线的对照（Related Work 归纳）

| 路线 | 代表 | 局限 | 本文差异 |
|------|------|------|----------|
| 多专家蒸馏 | Hoeller, Rudin, concurrent [28] | 专家分配与切换工程量大 | 单一生成器按感知在线产出参考，无需显式蒸馏管线 |
| 模仿 + 固定参考 | DeepMimic, BeyondMimic, SONIC | 测试时地形/朝向变化需新参考 | 在线生成 terrain-aware 参考 |
| 仿真跑酷扩散 | Xu et al. [30] | 主要仿真角色，真机全身感知不足 | G1 真机 onboard 感知 + 全身协调 |
| Co-diffusion 关节动作 | [8, 12] | 伪影可导致不稳定 locomotion | 生成器只出参考，RL tracker 保证物理执行 |
| 生成中间件恢复 | Heracles | 侧重 tracking↔recovery 桥接 | 本文侧重 **地形感知 locomotion 技能组合** |

### 量化结论（论文 Table I & Fig. 4）

- **在线生成 vs 固定参考：** 箱攀、跨栏、上楼梯三类任务上，Tracker+Gen 平均成功率显著高于仅用固定参考轨迹的 Tracker Only（尤其障碍高度/偏航角 OOD 时，如 70–80 cm 箱攀）。
- **闭环微调必要性：** 五类穿越任务上，耦合生成器后的 tracker **经微调** 成功率 consistently 高于未微调版本。
- **涌现行为：** 微调后出现局部绕行（参考不合适时跟踪器部分 override 参考仍达目标）；箱顶重定向、连续跨栏、训练未见地形组合间的动态风格切换。

### 真机演示要点

- 攀 **75 cm** 箱、多高度跨栏、楼梯上下、箱+栏+楼梯混合序列。
- 攀箱用膝与手支撑，跳下用手缓冲——与训练数据运动风格一致。
- 全 pipeline onboard：感知、生成、跟踪均在机器人上运行。

## 对 wiki 的映射

- 沉淀实体页：[Learning Whole-Body Humanoid Locomotion（42 篇栈 #27）](../../wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md)
- 项目页归档：[sources/sites/wholebody-locomotion.md](../sites/wholebody-locomotion.md)
- 42 篇栈策展摘录：[humanoid_rl_stack_27_learning_whole_body_humanoid_locomotion_via_moti.md](./humanoid_rl_stack_27_learning_whole_body_humanoid_locomotion_via_moti.md)
- 交叉补强：[扩散运动生成](../../wiki/methods/diffusion-motion-generation.md)、[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[Heracles](../../wiki/entities/paper-heracles-humanoid-diffusion.md)、[PHP 感知跑酷](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)

## 参考来源（原始）

- arXiv: <https://arxiv.org/abs/2604.17335>
- 项目页: <https://wholebodylocomotion.github.io/>
- 42 篇栈微信导读：[wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)

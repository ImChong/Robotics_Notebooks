# oasis_humanoid_loco_manip_2606_08548

> 来源归档（ingest · arXiv 全文精读）

- **标题：** OASIS: From Simulation Data Collection to Real-World Humanoid Loco-Manipulation
- **类型：** paper
- **作者：** Zehao Yu, Jiakun Zheng, Weiji Xie, Jiyuan Shi, Chenyun Zhang, Chenjia Bai†, Xuelong Li（中国电信人工智能研究院 TeleAI；复旦大学；华东理工大学；上海交通大学）
- **出处：** 2026-06 · arXiv:2606.08548
- **论文链接：** <https://arxiv.org/abs/2606.08548>
- **项目页：** <https://oasis-humanoid.github.io/>
- **代码：** <https://github.com/TeleHuman/OASIS>
- **入库日期：** 2026-06-14
- **一句话说明：** 真实物体照片 → Hunyuan3D+Qwen3-VL 自动建仿真场景 → VR 仿真遥操作采集状态轨迹 → 离线 Path-Tracing 视觉域随机化扩增 → Flow Matching 高层 + Teleopit 低层层级策略；在 Unitree G1 上 **零样本** 部署，**纯仿真数据** 在多数任务上 **匹配或超过** 等量真机 teleop。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2606.08548>
- **核心贡献：** 人形 loco-manipulation 演示数据在 **轨迹质量** 与 **可扩展性** 间两难：真机 teleop 质量高但空间/复位/硬件损耗成本高；OASIS 把 **整条数据管线搬进仿真**——从真实图自动生成物理资产、VR 实时 teleop 录状态、离线高保真渲染+域随机化扩增，再训练层级 visuomotor 策略。真机实验：零样本下 **仿真数据训练** 在多数任务成功率 **不低于甚至高于** 等量真机 teleop，主因仿真渲染覆盖更广的光照/环境变化。
- **对 wiki 的映射：**
  - [OASIS 论文实体](../../wiki/entities/paper-loco-manip-04-oasis.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)

### 2) Real-to-Sim 资产与场景自动构建（§3.2.1）

- **核心贡献：** 给定真实物体参考图 → **Hunyuan3D** 生成高分辨率纹理 mesh → **Qwen3-VL** 估计物理尺寸与材质类别 → 查表赋密度/摩擦/恢复系数并 **围绕预测值随机化** 物理参数。消除手工搭场景瓶颈，使仿真数据收集不依赖实体硬件。
- **对 wiki 的映射：**
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 3) 两阶段解耦数据采集（§3.2.2–3.2.3）

- **Stage A — 仿真 VR teleop：** PICO 4U 等便携 VR（头显+手柄+踝部 tracker）第一视角；**GMR** 重定向全身参考运动 → **Teleopit** RL 低层 WBC 驱动 Isaac Sim **Real-Time** 模式人机闭环；仅记录 **机器人+场景刚体运动学状态** 与 GMR 参考运动（非高保真图像）。
- **Stage B — 离线扩增渲染：** 回放状态轨迹，Isaac Sim **Path-Tracing** 离线渲染；随机化 **背景纹理、环境光强度/色温、相机外参**；单条演示扩为 **20** 个视觉环境（消融显示 15–20 趋于饱和）。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [Motion Retargeting (GMR)](../../wiki/methods/motion-retargeting-gmr.md)
  - [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)

### 4) 层级策略：Flow Matching 高层 + Teleopit 低层（§3.3）

- **高层：** Transformer action-chunking；条件 = 冻结 **CLIP** 文本 + 冻结 **DINOv2** 三视角图 + 最近 **H=2** 帧 **参考运动命令**（非带噪机器人状态，避免跟踪误差累积）；预测未来 **F=32** 帧、67-D 参考运动（TextOp 格式）；**Flow Matching** 训练，推理 10 步 Euler。
- **低层：** **Teleopit** 将参考运动转为 29-DoF 身体关节角；与 14-DoF 手合计 **43-DoF** 全身输出。
- **训练技巧：** **Curriculum rollout**——训练前 20% 仅用 GT 历史，之后 $p_{\text{rollout}}$ 线性升至 0.8，让模型适应自回归复合误差（w/o Rollout 四任务平均近崩）。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

### 5) 实验：效率、消融与 sim vs real 数据（§4）

- **采集效率（50 条成功轨迹/任务）：** OASIS 比真机 teleop 快 **1.15×–1.84×**（任务越难差距越大）；时间差主要来自真机 **场景复位与易碎物体**（擦屏任务曾损坏显示器）。
- **域随机化消融（零样本 10 次/任务）：** 关掉全部随机化平均成功率 **0.05**；全开 **0.83**；**光照** 单项贡献最大；纹理/相机外参互补。
- **数据对比（各 50 轨迹）：** 纯 OASIS 仿真数据在真机成功率 **可比或优于** 纯真机数据；**等量混合** 最优——仿真供视觉多样性，真机补交互/感知细节。
- **真机平台：** 29-DoF G1 + 7-DoF 三指灵巧手；头载 D435i + 双腕 D405；高层 25 Hz（RTX 4090），低层 50 Hz 执行 32-step chunk。
- **任务：** Place Cup in Box、Wipe Monitor、Lift Basket and Place Cup、Kneel and Wipe Under Table。
- **对 wiki 的映射：**
  - [LEGS](../../wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)（同为 G1 loco-manip 合成数据路线，LEGS 走 3DGS+VLA，OASIS 走 sim teleop+FM）
  - [VIRAL](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md)（RL 仿真数据 vs OASIS 的 teleop+渲染扩增）

### 6) 局限（§6）

- 仅随机化 **视觉**，轨迹多样性受操作员演示上限；全身状态扰动易破坏平衡。
- 生成资产几何/物理参数对复杂物体可能不准，接触丰富任务 sim2real 仍难。
- 依赖 **Hunyuan3D + Qwen3-VL + Isaac Sim + Teleopit + GMR** 栈与 G1 硬件配置。

## 数据效率摘要（Table 1）

| Task | OASIS (min) | Real (min) | Speedup |
|------|-------------|------------|---------|
| Place Cup in Box | 15.2 | 17.5 | 1.15× |
| Wipe Monitor | 19.1 | 26.8 | 1.40× |
| Lift Basket and Place Cup | 25.2 | 40.2 | 1.60× |
| Kneel and Wipe Under Table | 28.4 | 44.8 | 1.84× |

## 其他公开资料

- **项目页：** <https://oasis-humanoid.github.io/> — 归档见 [sources/sites/oasis-humanoid-github-io.md](../sites/oasis-humanoid-github-io.md)
- **代码仓库：** <https://github.com/TeleHuman/OASIS> — 归档见 [sources/repos/telehuman_oasis.md](../repos/telehuman_oasis.md)
- **Loco-Manip 8 篇策展：** [loco_manip_survey_04_oasis.md](loco_manip_survey_04_oasis.md)

## 当前提炼状态

- [x] 摘要与核心方法摘录（≥5 条）
- [x] wiki 页面映射
- [x] 项目页与代码链接

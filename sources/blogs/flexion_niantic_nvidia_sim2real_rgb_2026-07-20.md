# Niantic Spatial, Flexion, and NVIDIA: Closing the Sim2Real Gap for Humanoids

> 来源归档（blog / Flexion 官方，与 Niantic Spatial、NVIDIA 联合发布）

- **标题：** Niantic Spatial, Flexion, and NVIDIA: Closing the Sim2Real Gap for Humanoids
- **类型：** blog
- **作者：** Flexion Team
- **原始链接：** https://flexion.ai/news/niantic-spatial-flexion-and-nvidia-closing-the-sim2real-gap-for-humanoids
- **发表日期：** 2026-07-20
- **入库日期：** 2026-07-20
- **抓取方式：** 官方新闻页直接抓取（WebFetch）
- **合作方：** Niantic Spatial（Real2Sim 重建）、NVIDIA Isaac Sim / Isaac Lab（仿真与训练基础设施）、Flexion（Sim2Real RGB 导航策略与部署）
- **一句话说明：** 三方联合展示 **端到端 Real2Sim→Sim2Real 管线**：用 **360° 相机单次 walkthrough** 扫描部署现场 → **Niantic Spatial** 重建 **照片级 3DGS + 对齐碰撞 mesh** 并导出 **NuRec USDZ** → 在 **Isaac Lab** 中大规模并行 RL 训练 **纯 RGB 局部导航策略** → **零样本** 迁移到真机办公室；仿真评测中 **3DGS 重建场景训练的 RGB 策略** 是唯一达到或超过 **深度基线** 的 RGB 方案（Flexion 办公室 **97.8% vs 93.8%**；Niantic 办公室 **75.0% vs 70.9%**，各 1024 rollouts）。

## 核心摘录（归纳，非全文）

### 问题背景

- **RL 是人形基础技能（locomotion、navigation）最可扩展的训练方式**，但真机试错成本高、重置慢、易损硬件，故主流在仿真中训练。
- **关键问题：** 如何在机器人踏入现场之前，就知道 sim 训练的策略能在**该部署点**成立？答案是用**该地点的数字孪生**训练、测试与评测。
- **行业现状：** 大量 RL 策略仍在 **合成、无纹理** 环境中训练，感知输入被限制在 **深度图**（sim2real gap 相对可控）；**RGB 潜力更大**（成本低、FOV 宽、隐含几何+语义），但在仿真中训练 RGB 策略仍是开放问题。

### 联合管线愿景

1. 团队用标准 **RGB 相机** 扫描目标部署现场。
2. **Niantic Spatial** 重建高保真数字孪生。
3. 开发者在 **NVIDIA Isaac Sim / Isaac Lab** 开源栈中仿真训练。
4. **Flexion** 提供针对特定硬件调优的策略与部署软件，**零样本**迁移到真机与训练环境。

### Part 1：Niantic Spatial Real2Sim

| 环节 | 要点 |
|------|------|
| **采集** | 几分钟 **360° 相机** 单次 walkthrough；无需 LiDAR、三脚架站点或专用流程 |
| **视觉层** | 估计相机位姿 + 训练 **3D Gaussian Splat**；任意视点照片级 RGB；保留真实光照、材质与杂乱 |
| **物理层** | 同一重建派生 **碰撞 mesh**；深度用 **MVSAnywhere**（零样本 MVS）；白墙等低纹理区域比纯光度重建更平整 |
| **对齐** | 视觉 splat 与碰撞 mesh **同源重建、构造对齐**，避免「看见墙 vs 撞墙」不一致 |
| **导出** | splat + mesh 打包为 **NuRec volume USDZ**，**重力对齐、米制尺度、碰撞就绪**；直接载入 Isaac Sim / Isaac Lab；splat 走 RTX 渲染，mesh 作不可见物理代理 |

**局限（作者自述）：** 捕获时刻光照/反射/物体摆放被烘焙；静态场景（人/动态体后加）；未覆盖区域质量下降——但未阻止下文零样本迁移。

### Part 2：Flexion Sim2Real（局部 RGB 导航）

- **任务：** **局部导航** — 从当前位姿到附近目标、避障，**不依赖预建全局地图**；输入：机载相机 + 本体感知 + 机体系目标；输出速度指令，由 **预训练 locomotion 策略** 闭环执行。
- **训练：** NuRec 体积载入 Isaac Lab；单 GPU 上 **大规模并行 RL**；随机 spawn/目标；奖励到达、惩罚碰撞与跌倒。
- **Sim2Real 配方：** **仿真域随机化** + **大规模离线预训练图像编码器**（训练与机载部署共用），支撑百万级 rollout。
- **真机：** **纯仿真训练** 的 RGB 网络 **零样本** 部署；家具重排等场景变化下仍可用；名义导航性能 **与深度策略相当**。
- **深度传感器：** 实验用 **ZED X neural 模式**；RGB 在 **语义障碍（蓝垫）**、**细结构（三脚架/栏杆/线缆）**、**透明表面（玻璃门/窗）** 等深度难例上仍有效——训练重建中**未出现**这些具体实例，但仿真中与 **语义相近物体**（如窗户）的碰撞失败教会策略泛化。

### 仿真四策略对照（同 spawn/目标，1024 rollouts × 2 办公室场景）

| 策略 | 训练环境 | 传感器 |
|------|----------|--------|
| 基线 A | 生成无纹理导航 mesh | 深度 |
| 对照 B | 同上 | RGB |
| 对照 C | 合成纹理办公室 mesh | RGB |
| **本文** | **实际场地 3DGS 重建** | **RGB** |

**结果：** 两场景中 **仅 3DGS 重建 + RGB** 达到或超过深度基线；合成办公室与生成 mesh 上的 RGB 均落后，且场景越难差距越大。

### 未来方向（作者列举）

- 扩展采集硬件（iPhone、鱼眼）以扩数据集
- 重建上挂 **开放词汇语义** 标签
- **场景变换**（光照、门窗状态、家具布局）
- 仿真内 **端到端全任务评测**（指向 [Reflect v1.0](https://flexion.ai/news/flexion-reflect-v1.0) 长程自主方向）
- **VLM / Agent** 在相同重建场景中的语言条件行为

**商业含义：** 新场地策略部署从 **数月现场适配** 压缩到 **数日**。

## 开源 / 代码状态（项目页核查，2026-07-20）

| 组件 | 状态 |
|------|------|
| **Niantic Spatial 重建管线** | **未开源** — [nianticspatial.com](https://nianticspatial.com) 为企业服务与 Scaniverse 产品入口；公开 **SPZ**（Gaussian splat 文件格式）为开源格式，非完整重建/导出管线 |
| **NVIDIA Isaac Sim / Isaac Lab / NuRec** | **已开源/开放获取** — Isaac 栈与 NuRec volume 规范为 NVIDIA 官方仿真基础设施；见 [isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md) |
| **Flexion RGB 导航策略与部署栈** | **未开源** — Flexion 新闻页无 GitHub/权重链接；为零样本真机演示的产业栈 |

## 对 wiki 的映射

- [flexion-niantic-nvidia-rgb-sim2real-pipeline](../../wiki/entities/flexion-niantic-nvidia-rgb-sim2real-pipeline.md)（联合管线实体 + Mermaid）
- 交叉：[Sim2Real](../../wiki/concepts/sim2real.md)、[Flexion Reflect v1.0](../../wiki/entities/flexion-reflect-v1.md)、[GS-Playground](../../wiki/entities/gs-playground.md)、[LEGS](../../wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)、[SimFoundry](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md)、[Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Locomotion](../../wiki/tasks/locomotion.md)

## 可信度与使用边界

- **产业联合博客**，非 peer-reviewed；定量主要为作者自报仿真/真机演示。
- **场景专用 specialization：** 策略在**特定场地重建**中训练，泛化到新建筑需重新采集重建。
- **RGB 泛化边界：** 作者明确语义相似可泛化、**过远分布外物体** 仍困难；长期路线是 **大规模感知预训练 + 重建内微调**。
- **与 Reflect v1.0 关系：** 本文为 **局部导航 + Real2Sim 管线** 深度案例；Reflect 为更长程 mission 系统，文末列为下一步评测方向。

## Citation

```bibtex
@article{
    flexion2026nianticnvidiasim2real,
    author = {Flexion Team and Niantic Spatial and NVIDIA Robotics},
    title = {Niantic Spatial, Flexion, and NVIDIA: Closing the Sim2Real Gap for Humanoids},
    journal = {Flexion News},
    year = {2026},
    url = {https://flexion.ai/news/niantic-spatial-flexion-and-nvidia-closing-the-sim2real-gap-for-humanoids},
}
```

---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
venue: "Robotics: Science and Systems (RSS) 2024"
summary: "RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_ref_e064f89fc62c8df5da9f_universal-manipulation-interface-in-the.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Universal Manipulation Interface

**Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 128/564** 篇，分组 **Datasets**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 128/564 |
| 分组 | Datasets |
| 出处 | Robotics: Science and Systems (RSS) 2024 |
| 论文 | <https://www.roboticsproceedings.org/rss20/p045.html> |
| 子类 / 象限 | 数据采集接口 · 机器人示范与操作数据 · 跨机器人与多任务数据 · 不适用 |

## 核心机制（归纳）

> 注：以下机制与评测指标基于通读 RSS 2024 原文全文核对补充（[RSS 2024 原文](https://www.roboticsproceedings.org/rss20/p045.html)）。

**UMI（Universal Manipulation Interface）** 是一套无需真机即可在野外低成本采集高质量操作示教、并直接迁移到多种机械臂部署的数据采集与策略学习框架（软硬件全开源：<https://umi-gripper.github.io>）：

1. **手持式硬件示教接口（§III-A, HD1–HD6）**：
   - **极致轻量与低成本**：3D 打印扳机驱动平行夹爪（重 **780 g**，尺寸 **310 × 175 × 210 mm**，指爪行程 **80 mm**，BoM 成本 **\$73**）配单台腕部 **GoPro**（相机及配件 **\$298**），所有数据写入单个标准 MP4 文件（含内嵌 GPMF 格式 IMU  telemetry）。
   - **宽视场 + 隐式双目镜像（HD1–HD3）**：采用 **155° 超广角鱼眼镜头**并直接将原始鱼眼图送入策略（保留中心高分辨率、避免透视矫正拉伸边缘）；在夹爪两侧加装**物理侧镜**提供隐式双目深度线索，并在输入策略前对镜面裁剪区做**数字镜像翻转并互换左右镜面图像**以匹配主视图运动方向。
   - **IMU 辅助 SLAM 与连续夹爪追踪（HD4–HD6）**：基于 **ORB-SLAM3** 与 GoPro IMU 联合优化恢复带真实物理尺度的 6-DoF 末端轨迹（先建图再重定位 `map-then-localize` 使单双臂共享统一场景坐标系）；通过 fiducial marker 连续追踪**夹爪开合宽度**，配合柔性指垫实现隐式力控；离线依据目标机器人基座与运动学/动力学约束过滤不可达示教轨迹。
2. **硬件无关策略接口（§III-B, PD1–PD2）**：
   - **推理期延迟匹配（PD1）**：观测侧按实测延迟将图像（下采样至 10–20 Hz）与机械臂/夹爪状态线性插值对齐至相机时间戳 $t_{obs}$（双臂最近邻软同步偏差 $\le 1/60$ s）；动作侧根据硬件执行延迟（如臂 100 ms、爪 120 ms）与推理延迟提前发送指令，丢弃过期前缀仅执行 $t_{act}$ 之后动作。
   - **相对末端位姿与双臂互感知（PD2）**：以 **Diffusion Policy** 生成动作序列，所有动作与历史本体感知（horizon=2）均表示为相对当前推理帧 $t_0$ 夹爪位姿的 **$SE(3)$ 相对轨迹**，免除相机-基座标定且对移动底盘扰动天然鲁棒；双臂任务通过共享场景地图计算两夹爪间的相对位姿作为本体输入。

### 方法流程图

```mermaid
flowchart LR
  Hand["Handheld UMI Gripper (780g + 155 deg Fisheye GoPro + Side Mirrors)"] --> Rec["Single MP4 (RGB + GPMF IMU + Fiducial Width)"]
  Rec --> SLAM["Map-then-Localize ORB-SLAM3 + IMU (6.1 mm ATE) + Kinematic Filter"]
  SLAM --> Pre["Digital Mirror Flip & Swap + SE(3) Relative Trajectory"]
  Pre --> DP["Diffusion Policy (ViT / CNN Visual Enc + Proprio)"]
  DP -->|"Latency Matching (Obs Interp + Exec Lookahead)"| Robot["Cross-Embodiment Deployment (UR5 / Franka FR2, Bimanual or Mobile)"]
```

## 评测与指标

- **SLAM 追踪精度与采集吞吐（§VII, Fig. 11–12）**：
  - **SLAM 精度**（7 单臂 + 7 双臂 MoCap 基准）：单夹爪平均 ATE **位置 6.1 mm / 旋转 3.5°**；双夹爪相对位姿 RPE **位置 10.1 mm / 旋转 0.8°**。
  - **数据采集吞吐**（15 分钟含复位与故障）：Cup Arrangement 达 **111 CPH**（SpaceMouse 遥操作的 **3 倍以上**，为 **35 CPH** 遥操作 vs **231 CPH** 徒手）；Dynamic Tossing 达 **149 CPH**（徒手 237 CPH 的 64%，SpaceMouse 为 **0 CPH**）。
- **四项真机窄域能力任务与消融（§V, Fig. 8）**：
  - **Cup Arrangement**（305 demos，20 次评测）：UMI 达 **20/20 = 100%**；同一权重零样本部署至 **Franka Emika FR2** 达 **18/20 = 90%**。消融：`No Fisheye`（69° FoV）**55%**（11/20）、`Delta Action` **80%**（16/20）、`Absolute Action` **25%**（5/20）、`Raw Mirrors`（未数字翻转互换）**85%**（17/20）、`No Mirrors` **90%**（18/20）。
  - **Dynamic Tossing**（280 demos，6 类 YCB 物体抛投，120 次投掷）：UMI 达 **105/120 = 87.5%**；关闭延迟匹配（`No Latency Matching`）跌至 **69/120 = 57.5%**。
  - **Bimanual Cloth Folding**（250 demos，20 次评测）：UMI 达 **14/20 = 70%**；去掉双臂夹爪间相对位姿（`No Inter-gripper Pose`）跌至 **6/20 = 30%**。
  - **Dish Washing**（258 demos，7 步长程含加酱扰动恢复，20 次评测）：微调 CLIP **ViT-B/16** 达 **14/20 = 70%**；从零训练 ResNet-34 完全失败 **0/10 = 0%**。
- **野外零样本泛化实验（§VI, Fig. 9）**：
  - 3 人 12 人时在 **30 个真实场所**采集 **1,400 条**杯子整理数据，配 **CLIP ViT-L/14** 在未见户外咖啡桌（22/35）与流水黑石喷泉（21/25）评测：总成功率 **43/60 = 71.7%**（已见杯 28/40 = **70%**，未见杯 15/20 = **75%**）；仅用实验室窄域数据微调同一 ViT 骨干在野外成功率为 **0%**（0/20）。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Datasets** 分组逐条展开。
- 与依赖实体机器人与第三人称标定相机的 [DROID](paper-rcl-ref-aa9c28fbf4242ea03696-droid-a-large-scale-in-the-wild-robot-manipulati.md) 相比，UMI 无需在采数现场携带机械臂，数据与策略可零样本跨 UR5/Franka 迁移，但仅依赖腕部视角而无全局第三人称观测。
- 与 Dobb-E 等准静态手持工具相比，UMI 凭借视觉-惯性 SLAM 与双向延迟匹配首次支撑了动态抛投、双臂叠衣与长程刷盘任务。

## 结论

**UMI 通过“鱼眼+侧镜+IMU 腕部手持夹爪”和“延迟匹配+相对末端位姿策略接口”，在消除观测与执行本体差异的同时将野外数据采集效率提升至传统遥操作的 3 倍以上，并实现 71.7% 的野外零样本环境与物体泛化。**

- **物理与数字对齐缺一不可**：155° 原始鱼眼视野（55% → 100%）、数字翻转互换侧镜（85% → 100%）、相对末端轨迹（25%/80% → 100%）与双臂相对位姿（30% → 70%）均经严格消融证实显著决定成功率。
- **动态任务关键在时序补偿**：关闭推理期观测/动作延迟匹配会使动态抛投成功率从 87.5% 骤降至 57.5%。
- **大数据多样性不可由预训练替代**：在同样微调 CLIP ViT 的条件下，仅靠实验室窄域数据野外泛化率为 0%，而 30 场所 1,400 条野外数据可达成 71.7%。
- **开箱生态**：全套 3D 打印结构、BoM（约 \$371）、SLAM 与 Diffusion Policy 训练部署代码均已在 <https://umi-gripper.github.io> 开源。

## 常见误区或局限

1. **强依赖环境纹理进行视觉-惯性 SLAM**：纯白墙等低纹理场景易导致 ORB-SLAM3 建图与重定位退化，论文提出未来可结合第三人称相机与额外 fiducial 标记缓解。
2. **采数时未知部署机器人运动学限位**：离线阶段只能通过运动学过滤剔除不可达轨迹（Franka FR2 评测中 2 例失败即源于机械臂关节限位）。
3. **人手与夹爪自由度及重量差距**：780 g 夹爪重量与单自由度平行夹爪形态仍使采数效率约为徒手示教的 48%–64%，且无法直接迁移多指灵巧操作技能。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_ref_e064f89fc62c8df5da9f_universal-manipulation-interface-in-the.md`](../../sources/papers/rcl_awesome_wam_ref_e064f89fc62c8df5da9f_universal-manipulation-interface-in-the.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://www.roboticsproceedings.org/rss20/p045.html>

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://www.roboticsproceedings.org/rss20/p045.html)

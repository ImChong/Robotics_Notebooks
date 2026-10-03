---
type: entity
tags: [paper, humanoid, loco-manipulation, robot-free, egocentric-demonstration, imitation-learning, vla, sonic, unitree-g1, beihang, shanghai-innovation-institute, alibaba]
status: complete
updated: 2026-10-03
arxiv: "2609.38046"
venue: "arXiv 2026"
related:
  - ../tasks/loco-manipulation.md
  - ../methods/imitation-learning.md
  - ../methods/vla.md
  - ../concepts/whole-body-control.md
  - ../concepts/motion-retargeting.md
  - ./paper-halomi-humanoid-loco-manipulation.md
  - ./paper-notebook-humanoid-manipulation-interface.md
  - ./unitree-g1.md
sources:
  - ../../sources/papers/egoalign_arxiv_2609_38046.md
  - ../../sources/sites/egoalign-project.md
  - ../../sources/repos/egoalign.md
summary: "EgoAlign 将第一视角人类示范适配为 SONIC 可执行的动作与因果状态监督：PICO/GoPro 采集、实时 SONIC–MuJoCo 反馈、尺度对齐与闭环手部修正、最终状态重建；只用人类数据微调 π₀.₅，在 G1 上验证约 10 米搬运、未见目标导航与脚踩交互。官方代码与数据截至 2026-10-03 待发布。"
---

# EgoAlign：把第一视角人类示范变成可执行的人形移动操作监督

**EgoAlign**（*EgoAlign: Bridging the Human-Humanoid Gap for Long-Range Loco-Manipulation*，arXiv:2609.38046）面向一个具体缺口：人类视频和动作轨迹有任务经验，却没有人形控制器训练所需的机器人状态历史，且人机身材与控制器响应差会让机器人手部接触位置偏离示范。EgoAlign 在目标机器人模型、仿真器和 SONIC 连续全身接口上适配人体动作、重建配套状态，再仅用这些人类任务示范微调 VLA。作者来自北京航空航天大学、上海创智学院、四川大学与阿里巴巴集团。

## 一句话定义

一种无机器人任务示范的数据构造方法：将人类观测、经过机器人尺度与控制器适配的动作，以及因果回放得到的机器人状态配成策略监督。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 根据图像和指令预测动作的策略；本文采用 π₀.₅ 架构 |
| WBC | Whole-Body Control | 连续协调机器人全身动作的控制接口；本文用 SONIC |
| SMPL | Skinned Multi-Person Linear Model | 将穿戴追踪得到的人体动作表示为关节骨架 |
| G1 | Unitree G1 Humanoid | 本文物理部署与移动操作评测所用的人形机器人 |
| PICO | PICO 4 Ultra Head-Mounted Display | 记录第一视角并通过追踪器采集人体动作的头显 |

## 为什么重要

- **把人的经验转成机器人可用的监督。** 人体示范带有移动、观察和操作的自然连贯性，但不能直接作为机器人训练样本；该工作把动作适配和状态补齐放进监督数据构造。
- **将执行误差闭环带回示范。** 采集时通过 SONIC–MuJoCo 看到机器人对当前人体动作的响应，离线再用控制器回放修正手部交互几何，适应控制器实际做得到的动作。
- **评测跨越静态抓取。** G1 先移动约 10 米搬运篮筐，还测试不同目标位置与脚踩踏板；它把无机器人示范带入持续导航、携物和操作的连续任务中。
- **区分动作监督和状态监督。** 策略不直接输出关节位置；它预测 SONIC 的动作 token，由控制器结合当前本体状态历史解码为全身动作。

## 方法栈与流程总览

```
flowchart TB
  H["人体第一视角采集<br/>PICO 追踪 + 双 GoPro 视角"] --> F["SONIC–MuJoCo 实时反馈<br/>采集者修正被抑制的动作"]
  F --> A["尺度对齐 + 控制器闭环修正<br/>保留移动/下肢参考，修正手部几何"]
  A --> R["因果回放<br/>先记录当前机器人状态历史，再配同 tick 动作 token"]
  R --> T["只用人类示范微调 π₀.₅ VLA"]
  T --> D["部署到 G1<br/>VLA 预测 token，SONIC 用实时状态历史解码"]
```

主线是用仿真控制器反馈提高动作的可执行性，随后对最终修正轨迹重放，补上真实策略接口所需的状态监督；真机任务示范只用于最终评测，不参与此处的人类数据微调。

## 核心机制

### 1. 采集时让演示者看到机器人会怎么动

PICO 4 Ultra 头显与腕部、脚踝、腰部等五个追踪器采集人体运动；两台 GoPro 分别拍摄向下的近场操作视角和水平的远处导航视角。人体 SMPL 动作、抓取命令与两个视频流同步。采集时将动作送入 SONIC–MuJoCo 实时回放；如果机器人小步或抬腿动作被控制器压小，演示者可以放大动作后重录。

### 2. 几何对齐后再用控制器回放修正

第一步根据 G1 身体比例建立优化骨架，对肩、锁骨和肘部做受限姿态修正，使机器人手掌靠近人体示范中的交互目标，同时保留整体移动与下肢运动。第二步运行 SONIC–MuJoCo，根据执行后的手部残差更新目标并再次优化，最多两轮 refinement；未通过闭环轨迹验收的示范不会进入训练集。

### 3. 因果状态重建解决“人类视频没有机器人状态”

把最终修正动作交给 SONIC–MuJoCo 按 50 Hz 回放。每个控制时刻先取得机器人姿态、关节状态和控制器历史，再记录该时刻对应的动作 token，确保训练时是“当前观测/状态 → 当前动作”。状态重建得到 43 维机器人本体状态；动作监督由 64 维运动 token 与左右手命令构成。部署时 SONIC 用在线状态历史解码动作 token，而手部命令单独执行。

## 源码运行时序图

**不适用（截至 2026-10-03）：** 官方仓目前只含静态项目博客和媒体文件，README 声明方法代码未来发布；没有可辨识的训练、推理或部署入口，暂时无法据源码绘制运行时序。

## 工程实践

| 环节 | 论文设置与复现关注点 |
|------|--------------------|
| 目标平台 | Unitree G1，参考高度 1.25 m；用 SONIC 控制并以 MuJoCo 重放 |
| 人类采集 | 三位演示者；PICO 头显、五个追踪器、两台 GoPro；同步近场与远场视角 |
| 数据规模 | 200 条完整篮筐搬运、200 条导航、200 条脚踩交互示范；另有 100 条携篮导航示范；筛选后约保留 90% |
| 运动适配 | 先运动学尺度对齐，再最多两轮 controller-in-the-loop 修正；保留整体移动与下肢参考 |
| 监督标签 | 43 维本体状态；64 维 SONIC token + 双手命令；因果回放在 50 Hz 控制 tick 采集状态 |
| VLA 训练 | π₀.₅ 架构，以 PaliGemma-3B-PT-224 初始化视觉语言骨干；50,000 步训练，8 张 A800、global batch 256 |
| 推理执行 | 每块预测 50 步动作，在 50 Hz 执行后重新观测；SONIC 按实时本体历史继续闭环解码 |
| 采集效率 | 搬运示范的现场录制加重置时间从遥操作 130 秒降至人体示范 25 秒，约 5.2 倍 |

## 评测与实验证据

每个设置 20 次 G1 真机试验。物体搬运按接近、抬起、携带、放置四阶段计分；导航按抵达目标区域和最终对位计分；脚踩交互按接近踏板、有效踩下和稳定完成计分。多位置的未见目标相对对应训练目标沿地面轴偏移 0.5 m。

| 真机任务 | 设置 | 阶段完成分数 | 全任务成功率 |
|----------|------|---------------:|---------------:|
| 篮筐搬运约 10 m | Direct | 82.5 | 65% |
| 篮筐搬运约 10 m | Multi-seen | 71.3 | 40% |
| 篮筐搬运约 10 m | Multi-unseen | 75.0 | 40% |
| 导航 | Direct / seen / unseen | 72.5 / 77.5 / 75.0 | 65% / 70% / 65% |
| 脚踩交互 | Seen / unseen | 80.0 / 78.3 | 60% / 60% |

- **状态重建消融：** 训练数据缺少因果状态重建或使用零状态历史时，篮筐搬运 Direct 的第一个阶段成功率从完整方法的 20/20 降至 0/20。
- **尺度对齐消融：** 不对齐时仿真右手掌平均误差为 14.27 cm；仅运动学对齐后 6.48 cm；两轮控制器闭环 refinement 后为 1.65 cm。对应真机接近/抬起阶段成功率从 NoAlign 的 0%/0% 与 Round 0 的 50%/0% 提升到完整方法的 100%/90%。
- **双视角作用：** 移除水平远景相机后，多位置导航 full success 下降 25–30 个百分点；仅向下相机的近场覆盖不足以稳定找到远处目标。
- **现场采集速度：** 人体示范每条录制+重置 25 秒，遥操作为 130 秒；论文报告约 5.2× 吞吐提升，不含设备一次性搭建时间。

## 结论

**EgoAlign 的主要收益来自把控制器可执行性和因果状态配对纳入人类数据构造；单纯人体动作几何缩放不足以得到可训练的机器人监督。**

1. **先适配控制器，再生成训练状态。** 对最终轨迹做 SONIC–MuJoCo 回放，把修正后动作和它实际产生的当前机器人状态配对。
2. **闭环修正比单次几何对齐有明显收益。** 右手掌误差从运动学对齐的 6.48 cm 降至 1.65 cm；真机接近/抬起从 50%/0% 提升到 100%/90%。
3. **长程任务的远景信息不可省。** 去掉水平视角后，多位置导航成功率下降 25–30 个百分点；向下相机主要看近场操作。
4. **成功率要和任务阶段一起读。** 约 10 m 搬运在未见目标的全任务成功率为 40%，虽能成功拿起篮筐，但转向与抵达桌面的后续阶段仍会失败。
5. **“零样本真机”不是代码开源或全接触仿真。** 真机任务微调只用人类示范；但当前实现仍依赖 SONIC、G1 模型与仿真，回放不模拟真实篮筐/手足物体接触，方法代码和数据尚待发布。

## 与其他工作对比

| 路线 | 示范来源与适配 | 全身控制接口 | 本文任务覆盖 |
|------|----------------|----------------|--------------|
| HuMI | UMI 手部演示，任务空间策略 | 非通用控制器 | 未报告目标位置泛化与脚部交互 |
| BifrostUMI | UMI 示范与 learned tracker | tracker 代码尚未发布（论文时） | 未报告脚部交互 |
| EgoHumanoid | 第一视角人类示范，视角/动作对齐 | VLA 输出离散下肢动作 primitive | 未报告脚部交互 |
| **EgoAlign** | 双视角人体示范 + 几何缩放 + 控制器闭环修正 + 因果重放 | SONIC 连续全身接口 | 目标位置泛化、约 10 m 搬运、脚踩交互 |

和 [HALOMI](./paper-halomi-humanoid-loco-manipulation.md) 都从无机器人示范学习人形移动操作：HALOMI 重点是主动颈、BFM-Zero 潜空间跟踪与头手接口；EgoAlign 重点是控制器感知的人体动作适配及与动作 token 对齐的因果状态重建。二者都用 G1 真机测试，覆盖任务和数据接口不同。

## 局限与风险

- **接触物理被简化：** MuJoCo 回放只建模地面接触，不含真实篮筐及手/脚与物体接触；轻载、容错接触任务的成功不能直接外推到精细接触操作。
- **远距离物体搬运仍会掉在后半程：** Multi-seen / unseen 搬运全任务成功率均为 40%，主要失败出现在转向、越界或未能放到桌面。
- **当前实验规模有限：** 每项 20 次试验，示范量为数百条；没有证明对大量机器人形态或大规模复杂任务普遍成立。
- **实现与数据尚未发布：** 论文中多个优化阈值与筛选规则留待代码发布；截至入库日官方仓尚未提供实现和数据。
- **采集仍需成套设备：** 头显、五个追踪器、两台相机及对应标定/同步流程，可能成为复制成本。
- **依赖既有控制器：** EgoAlign 使用 SONIC 连续全身控制接口；换控制器时需要重新验证 token 语义、状态重建与回放闭环。

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md) — 任务方向、数据入口路线与相关工作索引
- [Imitation Learning](../methods/imitation-learning.md) — 人类示范如何成为机器人策略监督
- [Vision-Language-Action](../methods/vla.md) — 只用适配后的人体示范微调 VLA
- [Whole-Body Control](../concepts/whole-body-control.md) — SONIC 的连续全身动作执行接口
- [Motion Retargeting](../concepts/motion-retargeting.md) — 人体轨迹尺度与执行误差适配
- [HALOMI](./paper-halomi-humanoid-loco-manipulation.md) — 人类无机器人示范的移动操作对照
- [HuMI](./paper-notebook-humanoid-manipulation-interface.md) — robot-free UMI 示范接口对照
- [Unitree G1](./unitree-g1.md) — 真机评测平台

## 参考来源

- [arXiv HTML v2: EgoAlign](https://arxiv.org/html/2609.38046v2)
- [论文来源归档](../../sources/papers/egoalign_arxiv_2609_38046.md)
- [EgoAlign 官方项目页](../../sources/sites/egoalign-project.md)
- [EgoAlign 官方仓库与开源状态](../../sources/repos/egoalign.md)

## 推荐继续阅读

- [EgoAlign 项目页](https://lambdahumanoid.github.io/EgoAlign/)
- [arXiv:2609.38046 PDF](https://arxiv.org/pdf/2609.38046)
- [HALOMI](./paper-halomi-humanoid-loco-manipulation.md) — 主动感知与潜空间全身控制的无机器人示范路线
- [SONIC 项目与论文](https://arxiv.org/abs/2606.08640) — 连续全身控制接口


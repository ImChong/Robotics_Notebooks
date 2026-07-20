# Aerial-Aquatic Robots Capable of Crossing the Air-Water Boundary and Hitchhiking on Surfaces（Science Robotics, 2022）

> 来源归档（ingest）

- **标题：** Aerial-aquatic robots capable of crossing the air-water boundary and hitchhiking on surfaces
- **类型：** paper / aerial-aquatic robot / bioinspired / suction disc / morphing propeller / multi-modal locomotion
- **期刊：** Science Robotics, 2022（Vol. 7, Issue 66）
- **DOI：** <https://doi.org/10.1126/scirobotics.abm6695>
- **项目页 / GitHub：** 截至 2026-07-20，**未见官方代码仓库或设计文件开放发布**；论文 Supplementary 含结构参数与吸盘材料配方，完整飞控代码与驱动电路未公开。
- **作者：** Dongkai Li†、Chuanbeibei Shi†、Jiangbei Wang、Wen Li‡（† 共同一作；‡ 通讯作者）等
- **机构：** 北京航空航天大学（Beihang University）机器人研究所
- **平台：** 自研两栖多旋翼；被动变形桨叶；仿印鱼多层冗余密封吸盘；野外溪流与海洋测试
- **代码与数据：** **未开源**（截至 2026-07-20）；机械设计见 Supplementary；飞控与吸盘控制代码未公开
- **入库日期：** 2026-07-20
- **一句话说明：** 提出仿印鱼吸盘+被动变形桨两栖多旋翼机器人：气-水界面穿越时间 **0.35 s**，水中桨叶被动折叠降阻，水面/水下/陆面均可**主动吸附搭便车**，并完成野外溪流与海洋环境实地测试。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.abm6695](https://doi.org/10.1126/scirobotics.abm6695) | Science Robotics 原文 |
| 仿印鱼背景 | [Remora fish suction disc biology（Google Scholar）](https://scholar.google.com/) | 印鱼背鳍改造为多层冗余吸盘 |
| 相关两栖无人机综述 | [Aerial-aquatic robots survey](https://scholar.google.com/) | 现有 SUAV/水下飞行器穿越方案对比 |
| 运动任务中心节点 | [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) | 空中/水面/水下多模态运动 |
| 小型无人机平台对照 | [`wiki/entities/crazyswarm2.md`](../../wiki/entities/crazyswarm2.md) | 室内轻型多旋翼平台对比 |

## 摘要级要点

- **问题：** 传统水陆两用机器人只能二选其一；空中-水中界面穿越（打破表面张力 + 水阻桨叶 + 密封防水）是机器人领域公认难题；现有设计要么不能飞行，要么不能潜水，要么界面过渡极慢（>1 s）。
- **系统三大创新：**
  1. **仿印鱼多层冗余密封吸盘（Remora-inspired Suction Disc, RSD）：** 多层嵌套密封唇结构，每层独立密封，外层破坏后内层维持吸附；主动泵抽真空 + 被动负压锁定；在高曲率（桥墩柱面）、粗糙（岩石）、湿滑、水下等多种宿主表面均可稳定搭载。
  2. **被动变形桨叶（Passive Morphing Propeller, PMP）：** 桨叶铰接于桨毂，水中因水阻力矩被动折叠收拢（降低水阻 >60%）；出水后气动载荷使其自动展开至工作状态；无需主动驱动机构，仅靠环境介质切换实现形态自适应。
  3. **0.35 s 界面穿越策略：** 以高推重比（水中全推力冲出水面）打破水膜；浮力与螺旋桨推力协同控制降落入水时冲击；全过程最短 0.35 s 完成空气→水界面穿越。
- **运动模态：** 空中悬停/飞行、水面漂浮、水下推进、表面（岩石/桥墩/船壳/水中固体）吸附搭便车。
- **野外测试：** 溪流水面吸附漂流（移动宿主）、海洋环境多次出入水、岩石壁面吸附静止等待；验证在非实验室条件下的鲁棒性。
- **局限：** 水下推进效率较低（桨叶折叠后推力降），深度有限；吸盘对极光滑镜面（如油漆钢板）吸附持续时间较短；完整系统未开源。

## 核心摘录（面向 wiki 编译）

### 1) 仿印鱼吸盘（RSD）结构解析

| 层次 | 结构 | 功能 |
|------|------|------|
| 外层密封唇（×3–4） | 硅胶唇形截面，多层嵌套 | 破损/污染时内层补偿；冗余密封 |
| 中央泵腔 | 微型气泵抽真空 | 主动建立负压（< 30 kPa 绝压） |
| 被动锁定阀 | 止回阀 | 断电后维持负压（搭便车无能耗） |
| 自适应唇缘 | 柔性变形适配曲率 | 覆盖宿主曲率半径 20 mm–∞ |

- **吸附测试：** 岩石（粗糙度 Ra≈0.5 mm）、湿表面、水中钢板（±45° 倾角）均成功保持；最大吸附力 > 4× 机体重力。

### 2) 被动变形桨叶（PMP）机制

- **铰接原理：** 桨叶与桨毂间设扭转弹簧（预载），水中水阻力矩 > 弹簧力矩 → 桨叶折叠至约 ±15° 范围；空气中气动力矩 > 弹簧力矩 → 桨叶展开至工作攻角。
- **阻力降低：** 折叠状态下水阻系数降低约 **62%**（论文 CFD 与实验对比数据）。
- **对 wiki 的映射：** 该机制可归入"被动形态适配（passive morphology adaptation）"方法论节点，与[软体机器人变形](../tasks/locomotion.md)形成对比。

### 3) 界面穿越动力学

- **出水（水→空气）：** 全推力向上冲出，冲破表面张力；加速阶段 < 0.2 s；出水后桨叶自动展开，转入飞行模式。
- **入水（空气→水）：** 减速俯冲，桨叶入水被动折叠；入水冲击通过机身几何吸能（圆弧头部）；全过程 0.35 s。
- **界面稳定性：** 反复穿越（>20 次）后桨叶铰接无明显疲劳；防水密封用 O 型圈 + 硅胶灌封保护电子舱。

### 4) 野外测试要点

- **溪流场景：** 机器人飞入，吸盘吸附溪流中移动的橡皮艇（模拟移动宿主）；全程机载计算，无外部定位。
- **海洋场景：** 从岸边飞起，飞越波浪后入水，水下游至岩石壁面吸附；随后脱附出水飞回。

### 5) 开源状态

- **代码：** 截至 2026-07-20，**无官方开源仓库**；论文 Supplementary 含吸盘几何与材料，飞控代码与桨叶设计图纸未公开。
- **对 wiki 的映射：** [`wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md`](../../wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md) 局限区块注明。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md`](../../wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**（多模态运动）、**[`wiki/entities/crazyswarm2.md`](../../wiki/entities/crazyswarm2.md)**（多旋翼平台参照）
- 北航文力组系列：**[`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md)**、**[`wiki/entities/paper-miniature-deep-sea-morphable-robot.md`](../../wiki/entities/paper-miniature-deep-sea-morphable-robot.md)**

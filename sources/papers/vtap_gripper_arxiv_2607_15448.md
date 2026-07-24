# VTAP Gripper: Synergizing Fingertip Sensing and a Visuo-Tactile Active Palm for Dexterous In-Hand Manipulation（arXiv:2607.15448）

> 来源归档（ingest）

- **标题：** VTAP Gripper: Synergizing Fingertip Sensing and a Visuo-Tactile Active Palm for Dexterous In-Hand Manipulation
- **缩写 / 框架：** **VTAP Gripper**（Visuo-Tactile Active Palm）
- **类型：** paper / hardware / gripper / tactile-sensing / visuo-tactile / in-hand-manipulation / teleoperation / retargeting
- **arXiv：** <https://arxiv.org/abs/2607.15448>（HTML：<https://arxiv.org/html/2607.15448>；PDF：<https://arxiv.org/pdf/2607.15448>）
- **项目页：** <https://yuhao-zhou.com/vtap/>（`yuhochau.github.io/vtap/` 301 至此）— 归档见 [`sources/sites/yuhao-zhou-vtap.md`](../sites/yuhao-zhou-vtap.md)
- **代码：** **确认未开源**（截至 2026-07-24；项目页「Code」为 ViTacFormer 模板残留，非 VTAP 仓）
- **作者：** Yuhao Zhou、Sheeraz Athar†、Zhixian Hu†、Binghao Huang、Yunzhu Li、Juan Wachs、Yu She∗（† equal；∗ corresponding）
- **机构：** 普渡大学（Purdue）Edwardson 工业工程学院；哥伦比亚大学（Columbia）计算机科学系
- **会议 / 荣誉：** 录用 **IROS 2026**；2026 ASME SMRDC Finalist
- **入库日期：** 2026-07-24
- **一句话说明：** **13-DoF** 三指触觉反应夹爪：主动视触觉掌（LED 开关视觉/触觉）+ FlexiTac 指尖阵列 + 手势条件子空间重定向遥操作；在反应抓取、注射器手内操作、≥3 mm 手内 singulation、1 mm 公差 peg-in-hole 上验证「指–掌协同」可替代高 DoF 拟人手。

## 开源状态（步骤 2.5）

- **项目页核查（2026-07-24）：** <https://yuhao-zhou.com/vtap/> 提供论文与实验视频；**无** VTAP CAD / 固件 / 遥操作代码链接。页上 Code 按钮指向无关模板仓 ViTacFormer。
- **传感上游：** 指尖用 **FlexiTac**（[flexitac.github.io](https://flexitac.github.io/)，arXiv:2604.28156）——开源压阻阵列方案，**不等于** 本文夹爪开源。
- **结论：** **确认未开源**。Wiki 局限与工程实践须写明，避免读者误以为可按官方 README 复现。

## 摘录 1：问题与贡献（Abstract / §I）

- **痛点：** 平行夹爪难做抓后手内精操作；拟人手贵、重、难控；多数夹爪传感与驱动集中在手指，**掌–物接触与主动掌 DoF** 常被忽略；单模态（纯视觉或纯触觉）难同时覆盖远场定位与接触丰富反馈。
- **主张：** **VTAP Gripper** 用 **主动双模态掌 + 顺应可重构三指 + 指尖阵列** 做指–掌协同，并以 **分阶段、手势条件重定向** 弥合人手与异形三指的 embodiment gap，服务遥操作与下游学习数据采集。
- **贡献三点：** (i) 触觉反应夹爪与视触觉主动掌一体设计；(ii) 面向三指夹爪的手势条件遥操作重定向；(iii) 多样物体与任务上的系统验证。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-vtap-gripper.md`](../../wiki/entities/paper-vtap-gripper.md)；与 [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[触觉感知](../../wiki/concepts/tactile-sensing.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)、[Manipulation](../../wiki/tasks/manipulation.md) 互链。

## 摘录 2：机械与感知（§III-A / §III-B）

- **总 DoF：** 三指 × 4 轴 + 掌 1 轴线性驱动 = **13 DoF**。
- **手指：** Dynamixel 2XC430（径向/尺侧偏转 \(q_1\)、MCP \(q_2\)）+ 2XL430（内收/外展 \(q_3\)、PIP \(q_4\)）；TPU **Fin-Ray** 被动顺应；三指按 120° 扇区布置。
- **主动掌：** Actuonix L8-P-50，行程 **50 mm**；掌载 USB 相机（GC0307，FOV \(50^\circ\)）+ 可拆传感匣（丙烯酸 + 硅胶 + 镜面涂层）。
- **双模态切换：** LED **关** → 透明远场视觉；LED **开** → 漫反射照亮形变表面，成光学触觉；**无需机械换模 / 双目**。
- **指尖触觉：** FlexiTac \(32\times 12\) taxels，有效约 \(2\times 2\,\mathrm{mm}^2\) / 单元，贴片面积 \(66\times 25\,\mathrm{mm}^2\)。
- **工作空间（Monte Carlo）：** 指尖包络约 \(469\times 477\times 311\,\mathrm{mm}\)。

**对 wiki 的映射：** 实体页画硬件–感知–控制流程；强调「掌上视触切换」是硬件级阶段切换，与学习式门控融合互补。

## 摘录 3：手势条件重定向（§III-C）

- **难题：** 非拟人三指无关节一一对应；基座在几何中心而非腕；径向/尺侧偏转超出人手能力。
- **子空间：** 左手掌开合手势在 **cage / power / pinch** 三抓取先验间切换（改写各指 \(q_1\)）；并固定内收/外展以消歧。
- **中间系 \(\mathcal{I}\)：** 对人腕 VR 系 \(\mathcal{W}\) 施常值 \(SE(3)\)，对齐夹爪基座语义。
- **优化：** \(\mathcal{L}=\lambda_1\mathcal{L}_{\mathrm{pos}}+\lambda_2\mathcal{L}_{\mathrm{rot}}+\mathcal{L}_{\mathrm{vel}}\)（Huber）；求 \(\mathbf{q}_t^*\) 受关节限位。
- **Singulation 模式：** 拇指–食指距离/夹角映射到两主动指 \(q_3\)（等幅反向）与 \(q_2\) 微调。
- **控制环：** Quest 3 ~25 Hz 求解 → 插值到 100 Hz；夹爪电流 PD；UR5e RTDE 位置控制。

**对 wiki 的映射：** 与 [运动重定向](../../wiki/concepts/motion-retargeting.md) 对照——本文是 **末端夹爪遥操作重定向**，非全身 MoCap→人形。

## 摘录 4：实验要点（§IV–§V）

| 任务 | 主读数 |
|------|--------|
| 触觉反应抓取（9 物 ×10 试） | 总成功率 **93.3%**；Drill **5/10**（质量偏心易滑） |
| 手内球重定向 | \(x/y\) 约 \(\pm 15^\circ\)，\(z\) 约 \(\pm 20^\circ\) |
| 注射器重定向 + 柱塞（20 次遥操作） | **65%**（13/20）；均时 **33.4±5.43 s**；失败多因 VR 遮挡误释 pinch |
| 手内 singulation（5 类物） | 可至约 **3 mm** 直径；触觉图案验证「多接触→单物体」 |
| 自主 peg-in-hole（10 次） | **70%**；孔定位靠触觉边缘+圆拟合；公差 **1 mm** |

**对 wiki 的映射：** 结论节写清「硬件协同 vs 拟人手」选型读法与 VR 跟踪失败模式。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-vtap-gripper.md`**（流程总览；源码时序图标不适用）。
- 新建 **`sources/sites/yuhao-zhou-vtap.md`**。
- 交叉更新视触觉融合、触觉感知、接触丰富操作、Manipulation；可选回链 GelSlim（视觉触觉对照）。

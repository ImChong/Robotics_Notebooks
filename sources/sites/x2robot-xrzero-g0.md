# XRZero-G0 Project Page（自变量机器人）

> 来源归档

- **标题：** XRZero-G0: Pushing the Frontier of Dexterous Robotic Manipulation with Interfaces, Quality and Ratios
- **类型：** site / project page + technical report（无本体数采系统：硬件 + 质检流水线 + 数据配比实验）
- **URL：** <https://x2robot.com/x2go>（英文：<https://x2robot.com/en/x2go>）
- **官网技术博客时间线条目：** <https://x2robot.com/blog> — 条目「XRZero-G0」，日期 **2026.06.10**，链向 `/x2go`；描述原文："XRZero-G0 is an embodied data acquisition and strategy learning system designed through deep collaboration between hardware and software. It aims to overcome the fundamental bottleneck of acquiring high-quality, motion-aligned demonstration data in the field of dexterous robot operation."
- **论文：** arXiv:[2604.13001](https://arxiv.org/abs/2604.13001)（Comments: *Technical Report*；v1 2026-04-14，v2 2026-04-16；cs.RO；CC BY 4.0）
- **代码：** <https://github.com/X-Square-Robot/XRZero-G0>（仅 README + 配图，见下表）
- **数据集：** <https://huggingface.co/datasets/x-square-robot/XRZero-G0-3K>
- **商用产品页（推测对应）：** <https://x2robot.com/pages/quanxtazero>（QUANXTA Zero 系列「无本体数采」，其中 **Zero-G0 = UMI-VR 版**）
- **机构：** 自变量机器人（X Square Robot）
- **作者：** James Wang、Primo Pu、Zephyr Fung 等 22 人（BibTeX 末三位 Qian Wang、Roy Gan、Hao Wang）；论文标注 † Project Lead / ‡ Correspondence，HTML 版未能还原对应到人
- **日期：** 论文 2026-04-14；项目页标 *April 2026*；官网博客时间线 2026-06-10；HF 数据集创建 2026-05-16；GitHub 最后提交 2026-05-21
- **入库日期：** 2026-10-09
- **一句话说明：** 背包 + PICO 4 VR 头显 + 两种手持夹爪（H 型按压 / G 型手指驱动）做无本体双臂采数；用「采集→质检→训练→评测」闭环把有效率做到 85%，并报告 robot-free : 真机 = 10:1 时可追平 500 条纯真机基线。页面为 Next.js 前端渲染，正文取自 RSC payload。

## 开源核查（2026-10-09）

| 入口 | 状态 |
|------|------|
| Homepage | 已挂链 — <https://x2robot.com/x2go>；页内互链 arXiv abs/PDF、GitHub、HF 数据集 |
| Code / GitHub | **占位**：`git clone` 仅 `README.md`、`imgs/head.png`、`imgs/head.pdf`；14 次提交（2026-04-11 → 2026-05-21）全是 README/配图。README 挂 MIT 徽章但 **无 LICENSE 文件**；无采集端软件、质检流水线、训练脚本 |
| Weights | **未列**：项目页、README、论文均无 checkpoint 链接 |
| Data | **部分公开**：HF `x-square-robot/XRZero-G0-3K` 公开且未设 gate；**无 dataset card**（README 404，未声明许可证）。LeRobot `codebase_version: v3.0` 格式，**20** 个任务子目录、合计 **3,697** episode / **1,359,573** 帧（逐个 `meta/info.json` 加总）。与论文 G0-Dataset「>2,000 h / 3,000 任务」相比只是很小的子集 |
| Hardware | **未开源**：无 CAD / BOM / 固件；以 QUANXTA Zero-G0 商品形态出售（对应关系为推测，见下） |

**开放程度：部分开源（仅数据子集）。** 代码与权重未发布，硬件闭源；论文数字无法用官方代码复现。

### HF 数据集结构速记（2026-10-09 读取）

- 相机：`observation.images.faceImg` 1280×720；`leftImg` / `rightImg` 640×480（h264）。
- `action` 与 `observation.state` 均为 **14 维**：左右各 `pos_x/y/z`、`rot_x/y/z`、`gripper`——即双手末端位姿 + 夹爪开合，**无关节角、无触觉**。旋转参数化未说明。
- `robot_type: null`；`meta` 写 `fps: 30`，但各视频流 `video.fps: 20`——同一文件内帧率不一致，使用前需自行核对时间戳。
- 任务名示例：`fold_towel`、`arrange_1_flowers`（746 episode，最多）、`open_the_lock`、`Pour_the_kettle_water_into_the_pot`、`soak_and_clean_spilled_liquid_with_sponge` 等。
- 名称中「3K」的含义官方未说明（3,697 episode 或「3,000 任务」均可能，**推测**，不作结论）。

## 页面与论文内容要点

- **三个瓶颈 → 三个模块：** Interfaces（接口）、Quality（质量）、Ratios（配比）。
- **接口：** PICO 4 头显 inside-out 6-DoF 跟踪（项目页写 **≤4 mm**；论文 Table 1 记 4 mm，对比 UMI 10 mm、FastUMI 8 mm）；头显可调 RGB 相机作俯视主视角 + 双腕相机 = **3 视角**；VR 手柄刚性固定在两种自研夹爪上——**H 型**按压驱动（大尺度抓取）、**G 型**手指驱动（精细操作）；两夹爪间距按目标双臂基线标定；背包边缘计算单元对齐语言指令、6-DoF 手柄轨迹与 **30 Hz** 多视角视频。
- **质检四步：** ①图像质量评估丢模糊帧、静止帧降采样；②按目标 URDF 重定向到末端空间，IK 过滤关节限位 / 奇异 / 自碰撞；③每类任务抽样在目标双臂上 **开环物理回放**，能完成才算有效；④长轨迹切子任务并标注物体与关键帧。有效率「up to **85%**」（分母定义未给）。
- **配比思路：** 大量 robot-free 数据预训练视觉-语义与空间表征，少量真机数据微调作「kinematic anchor」（Few-Shot Physical Anchoring）。
- **数据集：** G0-Dataset **>2,000 h**、**3,000** 个长尾任务；峰值采集 **93.2 episode/h**。
- **实验平台：** 双臂 **CX001**（多关节、偏灵巧）与 **EX001**（大负载、大工作空间）；策略 Wall-OSS、π₀、π₀.₅。
- **RQ1 效率（对主从遥操作）：** 平均单条用时 简单 35→15 s（**2.33×**）、中等 75→40 s（**1.88×**）、困难 120→70 s（**1.71×**）；标准 VR 遥操作对照只在图中。
- **RQ2 回放：** 经 IK 在 CX001 / EX001 上做 1:1 空间回放，正文为定性结论。
- **RQ3 纯 robot-free：** 抓葡萄/茄子/香蕉，300→500 episode 成功率随量线性上升，Wall-OSS 在茄子与香蕉 500 条时 **75.0%**；双臂插花扩到 **2,000** 条，Wall-OSS 在 H=0.4 m **70%**、未见 H=0.45 m **60%**。
- **RQ4 配比：** 基线 500 条真机；1:1（500 真机 + 500 robot-free）把插花 Wall-OSS 从 **50% → 75%**；10:1（500 robot-free + **50** 真机）叠毛巾 **87.5%** 与基线相同、抓香蕉 **75.0%** 与基线相同。任务共 5 个（抓香蕉、抓葡萄、叠毛巾、往电饭煲加香肠、插花）。
- **成本：** robot-free 采集成本约为真机遥操作的 **1/20**，依据为「设备维护、平台开发、人力约束」的综合估计，**无分项明细**。
- **未来工作（自述）：** 背包计算单元偏重限制超长时采集，计划轻量化；继续压真机数据下限；扩展到全身移动操作。

## 需注意的口径问题

- 摘要与项目页说 2,000 h 数据集「enables **zero-shot** cross-embodiment transfer」，但 Fig. 8 图注写跨本体 rollout 的策略来自 **1:1 混合**（含 500 条真机数据）训练；「zero-shot」应理解为不针对第二台本体单独采数，而非完全不用真机数据（**推测**的读法，原文未澄清）。
- 「≤4 mm」「85%」「1/20 成本」均为作者自报，无误差条或原始数据下载。
- 论文 §5.1 称夹爪「detailed in Section 4」，实际在 §3.1，属编辑瑕疵。

## 商用产品对应（QUANXTA Zero 系列，推测）

官网「产品 → 无本体数采」页 <https://x2robot.com/pages/quanxtazero> 介绍 QUANXTA Zero 系列：**G0**（VR 头显 + 背包 + 双夹爪，「UMI-VR 版」）、**G1**（头环 + 双夹爪，「UMI-VIO 版」）、**E0**（单头环，「Ego 版」）。该页前端命名空间为 `xrzero-ns`，且 G0 形态与论文硬件一致，因此**推测** QUANXTA Zero-G0 即 XRZero-G0 的商品化版本（官方未明确写出二者关系）。页上 G0 参数：适配器供电；内置 512 GB SSD；Wi-Fi 2.4/5 GHz + 设备间有线传输；双目 RGB ×1、单目 RGB ×2、定位相机 ×4；夹爪相机 640×480@30 fps，头部双目 RGB 1280×720@30 fps；输出三摄 RGB、空间定位轨迹、语言指令、夹爪位姿与开合（「毫米级精度」）。系列级宣称：传感器同步 **<1 ms**、「1000 条无本体 + 100 条真机 ≈ 1000 条真机」、效率 **2.33×**。页尾免责声明：参数仅供参考，以合同为准。

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/xrzero-g0.md`](../../wiki/entities/xrzero-g0.md)**
- 同机构对照：[TwinDEX](../../wiki/entities/twindex.md)（2026-09，三指外骨骼 + 同构手，未开源）——二者是不同系统
- 交叉：[灵巧操作数据采集指南](../../wiki/queries/dexterous-data-collection-guide.md)、[具身数据采集五路线](../../wiki/queries/embodied-data-collection-five-routes-landscape.md)、[HandUMI](../../wiki/entities/handumi.md)、[UMI-FT](../../wiki/entities/paper-umi-ft.md)

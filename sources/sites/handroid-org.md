# handroid.org（Handroid 项目页）

- **标题：** Handroid: Bridging Dexterous Hand and Humanoid — 官方项目页
- **类型：** site / project-page
- **URL：** <https://handroid.org/>
- **入库日期：** 2026-07-20
- **配套论文：** [Handroid（arXiv:2607.16187）](https://arxiv.org/abs/2607.16187) — 归档见 [`sources/papers/handroid_arxiv_2607_16187.md`](../papers/handroid_arxiv_2607_16187.md)
- **配套 CAD：** <https://cad.onshape.com/documents/d3de21915f3c9cacc1887cf3/w/dc7c7b68235fdbb205f27505/e/8673167885a4e16e9b6c2791>
- **配套 BOM：** <https://docs.google.com/spreadsheets/d/1ml2pJ9iSiDhcNiEPnRkoHqwzarEjfeZ8KSDNHGFpFh4/edit?usp=sharing>
- **代码：** 截至入库日项目页 **Code 按钮为占位链（`href="#"`）**，**未列出 GitHub / Hugging Face 等代码仓库**；GitHub 组织 [`robot-handroid`](https://github.com/robot-handroid) 仅含静态站镜像 [`robot-handroid.github.io`](https://github.com/robot-handroid/robot-handroid.github.io)。

## 一句话摘要

UNC Chapel Hill × Stanford 团队 **Handroid**：桌面级 **27-DoF** 机电一体平台，可在 **灵巧手（20 指 DoF）** 与 **桌面人形（含 12-DoF 下肢）** 两种形态间 **滑轨重配置**；全 **3D 打印**、模块化；项目页聚合论文、Onshape CAD、BOM 与多段真机演示视频。

## 公开信息要点（截至入库日）

- **作者与机构：** Ruogu Li、Chenyang Ma、Sikai Li、Zhenyu Wei、Yunchao Yao（**UNC Chapel Hill**）；Haochen Shi、C. Karen Liu、Shuran Song（**Stanford University**）；通讯作者 Mingyu Ding（UNC）。
- **形态规格（摘要 / 项目页）：** 高 **0.33 m**、重 **2.05 kg**；**27 主动 DoF**；模块 I–V 在两种形态间分别映射五指 / 头+双臂+双腿，模块 VI–VII 为躯干与髋；关节 **9、26** 为棱柱滑动机构驱动形态切换。
- **设计卖点（页面分区）：**
  - **Dexterous hand embodiment**：类人手 DoF 布局，支持遥操作、策略学习与精细操作。
  - **Shared mechanical grammar**：指、臂、腿、底座复用同一紧凑模块族。
  - **Humanoid embodiment**：同一套机构可做人形行走、蹲起、俯卧撑/引体向上与 loco-manipulation。
- **电子与控制（论文 / 页内 Abstract 区）：** 定制 **40×80 mm** 堆叠主板；**ESP32-S3** + **Dynamixel TTL** 总线；**Wi-Fi** 状态流；**IMU**（躯干 + 指尖/足端）；支持电池或 **PD 140 W** 外供。
- **演示任务（Demos 区，节选）：**
  - **手形态：** Diffusion Policy 抓取、仿真 **PPO** 掌内立方体重定向（30 Hz）、指遥操作、倒水、叠杯、摘手套、双物体抓取、薄纸抓取等。
  - **人形态：** 速度指令行走/转向/蹲起、俯卧撑、引体向上、pick-and-place。
  - **长时程：** 形态切换 → 人形行走与交互 → 与 **Franka** 臂对接 → 灵巧 pick-and-place（页面 **Long Horizon** 视频）。
- **资源按钮：** Paper（arXiv）✓；CAD（Onshape）✓；BOM（Google Sheets）✓；**Code ✗（占位）**。

## 源码 / 数据开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| **机械 CAD** | **已发布** | Onshape 公开文档（见上链） |
| **BOM / 装配** | **已发布** | Google Sheets BOM |
| **控制 / 学习代码** | **待发布** | 项目页 Code 无有效 URL；论文写 "open-source" 但截至入库日未见官方训练/部署仓库 |
| **项目页源码** | **部分** | `robot-handroid/robot-handroid.github.io` 仅为静态站 |

## 为何值得保留

- **形态复用范式**：把「手」与「桌面人形」做成 **同一 27-DoF 机体的两种配置**，区别于固定形态灵巧手或 mini humanoid 平台（如 ToddlerBot、LEAP Hand）。
- **学习栈跨度大**：单平台覆盖 **Apple Vision Pro 遥操作**、**Diffusion Policy** 抓取、**RL 掌内操作**、**ZMP+LIPM 步态 + RL 跟踪**、**无参考速度 RL**、**Viser 关键帧编辑** 与 **跨形态长时程任务**。
- **一手入口**：论文、CAD、BOM、演示视频均从该页链出，适合 curator 跟踪 **代码何时落地**。

## 关联资料

- 论文归档：[`sources/papers/handroid_arxiv_2607_16187.md`](../papers/handroid_arxiv_2607_16187.md)
- GitHub 静态站：[`robot-handroid.github.io`](https://github.com/robot-handroid/robot-handroid.github.io)

## 对 wiki 的映射

- [Handroid](../../wiki/entities/handroid.md)

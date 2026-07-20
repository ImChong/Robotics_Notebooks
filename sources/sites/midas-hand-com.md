# midas-hand.com（MIDAS Hand 项目页）

- **标题：** MIDAS Hand — Modular low-Impedance Direct-drive Anthropomorphic Sensing Hand
- **类型：** site / project-page
- **URL：** <https://midas-hand.com>
- **入库日期：** 2026-07-20
- **配套论文：** [MIDAS Hand（arXiv:2607.14487）](https://arxiv.org/abs/2607.14487) — 归档见 [`sources/papers/midas_hand_arxiv_2607_14487.md`](../papers/midas_hand_arxiv_2607_14487.md)
- **配套代码：** <https://github.com/midas-hand-org>
- **代码：** <https://github.com/midas-hand-org>（已开源：API / MuJoCo / Retargeter / Teleop / PCB 文档 / 项目站源码）
- **联系：** contact@midas-hand.com

## 一句话摘要

UCLA **Dennis Hong** 组开源 **直驱低阻抗仿人触觉灵巧手** 官方站点：聚合 **BOM/CAD/装配/软件** 文档与演示视频；公开 **16 总 DoF / 13 主动 DoF、283 三轴 taxel、~700 g、BOM ~$3k、<3 h 装配** 等关键指标。

## 公开信息要点（截至入库日）

- **作者：** Alvin Zhu*、Mingzhang Zhu*、Beom Jun Kim*、Quanyou Wang*、Jose Victor S. H. Ramos*、Dennis Hong（* equal contribution；UCLA）。
- **开源状态（项目页核查）：** **已开源** — 站点分区 **Parts / CAD / Assembly / Software / Citation** 均链出可下载资源；Software 页列出四个 Python 仓库 + `midas_hand_communication_board` PCB 文档 + `midas-hand-org.github.io` 静态站源码。
- **硬件亮点：**
  - **Parts**：BOM、供应商、替代件、加工与工具清单。
  - **CAD**：Onshape 源文件、整手 STEP、3MF 打印板、URDF、版本说明。
  - **Assembly**：装配顺序、子装配、视频、标定与排障。
- **软件栈（Software 页）：**
  - `midas_hand_api` — Dynamixel + Paxini GEN3 触觉 Python 接口、homing/标定。
  - `midas_hand_mujoco` — MJCF/URDF/STL、交互/无头仿真入口。
  - `midas_hand_retargeter` — dex-retargeting 封装、landmark 适配与被动耦合模式。
  - `midas_hand_teleop` — MediaPipe 摄像头 → 打印/MuJoCo/真机后端。
- **演示分区：** 灵巧操作、触觉传感、rollout、遥操作、硬件重复性测试、MuJoCo 同步仿真。

## 为何值得保留

- **一手开源核查锚点**：BOM/CAD/四仓库/PCB 均可从该页直达，适合判断「全栈开放」边界。
- **触觉 + 直驱 + 低成本**：相对 LEAP（无触觉）与腱驱动开源手，MIDAS 把 **DTA 触觉** 作为默认模态发布。
- **学习数据采集闭环**：重定向 + 遥操作 + 同步关节/触觉流，服务模仿学习与 sim2real 实验设计。

## 关联资料

- 论文归档：[`sources/papers/midas_hand_arxiv_2607_14487.md`](../papers/midas_hand_arxiv_2607_14487.md)
- 代码归档：[`sources/repos/midas-hand-org.md`](../repos/midas-hand-org.md)

## 对 wiki 的映射

- [MIDAS Hand](../../wiki/entities/midas-hand.md)

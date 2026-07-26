# Yale OpenHand Model F3（型号页）

- **标题：** Model F3 — Forces-for-Free Hand（OpenHand）
- **类型：** site / model-page
- **URL：** <https://www.eng.yale.edu/grablab/openhand/model_f3.html>
- **入库日期：** 2026-07-26
- **所属项目页：** <https://www.eng.yale.edu/grablab/openhand/>
- **CAD：** <https://github.com/grablab/openhand-hardware/tree/master/model%20f3%20(forces-for-free%20hand)>
- **装配指南：** 仓库内 `Model F3 Assembly Guide 1.0.pdf`（站点链 `model f3/Model F3 1.0.pdf`，Rev 1.0，标注 12/15/24）
- **配套论文：** 页内写 **[paper under review]**（视觉力估计系统）；截至 **2026-07-26** 未见公开 DOI / arXiv
- **许可：** CC BY-NC 3.0（同 OpenHand 项目）

## 一句话摘要

**Model F3** 是 OpenHand **Model T42** 的 flexure–flexure 改编版：优化连杆几何与腱路由以降低摩擦、避免指尖接触奇异，配合 **腕部相机观测夹爪形变** 估计接触力，从而在无力/力矩传感器条件下做力控擦拭、插销与书法等任务。

## 公开规格（Performance）

| 项 | 值（型号页） |
|----|--------------|
| 执行器 | **2× Dynamixel XM-430-W350-R** |
| 基座高度 | 55–80 mm |
| 基座直径 | 90–200 mm |
| 质量 | **400 g** |
| 保持力 | **10 N** |
| Capabilities 视频 | Coming Soon（截至入库日） |

## 设计要点（About）

- 基于 T42 **flexure–flexure** 手指变体。
- **连杆长度与角度** 调整：避免指尖接触时的运动学奇异。
- **腱路由与电机位姿** 优化：显著降低腱摩擦 → 更利于由形变反推力。
- 与「视觉力估计系统」联用：可完成 force-controlled wiping、peg-insertion、calligraphy writing，**无需 FT 传感器**。

## 源码开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| **CAD / STL / SolidWorks** | **已开源** | `openhand-hardware` 目录 `model f3 (forces-for-free hand)/`（含 `stl/`、`sldprt/`、手指装配） |
| **装配 PDF** | **已开源** | Rev 1.0（2024-12-15） |
| **视觉力估代码 / 论文** | **未公开 / 审稿中** | 页内 paper under review；勿写成可复现力估全栈 |
| **站点 GitHub 深链** | **可能过时** | HTML 曾指向 `forces for free (F3) hand` 路径；仓库实际目录名为 `model f3 (forces-for-free hand)`（以 README / API 为准） |

## 为何值得保留

- OpenHand 族中明确面向 **免 FT、视觉估力接触任务** 的型号入口。
- 规格与装配材料完整，可与 T42 / EN02-OP 等低成本末端对照选型。
- 审稿论文状态需跟踪，避免 wiki 误写「力估已开源」。

## 关联资料

- 项目总页：[`yale-grablab-openhand.md`](yale-grablab-openhand.md)
- CAD：[`../repos/openhand-hardware.md`](../repos/openhand-hardware.md)

## 对 wiki 的映射

- [Yale OpenHand](../../wiki/entities/yale-openhand.md)（Model F3 专节）

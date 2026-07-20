# bottle101.github.io/AMI-EV（AMI-EV 项目页）

- **标题：** Microsaccade-inspired Event Camera for Robotics — 官方项目页
- **类型：** site / project-page
- **URL：** <https://bottle101.github.io/AMI-EV/>
- **入库日期：** 2026-07-20
- **配套论文：** [Science Robotics 2024](https://doi.org/10.1126/scirobotics.adj8124) · [arXiv:2405.17769](https://arxiv.org/abs/2405.17769) — 归档见 [`sources/papers/microsaccade_inspired_event_camera_scirobotics_2024.md`](../papers/microsaccade_inspired_event_camera_scirobotics_2024.md)

## 一句话摘要

ZJU + UMD 团队的 **AMI-EV** 官方站点：展示旋转楔形棱镜主动微扫视增强事件相机的 **系统原理、性能基准对比、开源仿真器与工具链**，以及低层特征跟踪与高层人体姿态估计两类任务的真实场景演示。

## 公开信息要点（截至入库日）

- **机构：** 浙江大学（Fei Gao、Chao Xu 课题组）、马里兰大学（Cornelia Fermüller、Yiannis Aloimonos）、香港科技大学（Shaojie Shen）
- **项目亮点（主页内容）：**
  - 系统原理动画：楔形棱镜旋转 → 光路偏折 → 持续事件生成 → 几何补偿 → 稳定输出
  - 三类场景对比（结构化、非结构化、极端光照）：图/视频对比 AMI-EV vs 灰度相机 vs 标准事件相机
  - 低层任务：特征检测与跟踪在三类光照条件下的对比结果
  - 高层任务：人体检测与姿态估计在强逆光场景下的成功演示
  - **Open-sourced Simulator** 专区：仿真器说明 + 示例输出（3D 场景渲染 + 数据集翻译器）

## 源码开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 发布渠道 | **Zenodo** | [10542425](https://zenodo.org/records/10542425)（镜像 [8157775](https://zenodo.org/records/8157775)）：`Hardware.zip` / `Software.zip` / `Simulator.zip` / `Dataset.zip` |
| 硬件设计 | **已开源（Zenodo）** | Hardware.zip；非独立 GitHub 应用仓 |
| 软件（标定/补偿） | **已开源（Zenodo）** | Software.zip |
| 仿真平台 | **已开源（Zenodo）** | Simulator.zip |
| 数据集 | **已开源（Zenodo）** | Dataset.zip；Translator 见软件包 |
| 项目页 GitHub | **模板仓 only** | 页脚 Academic-project-page-template，**不是** AMI-EV 源码仓 |

## 为何值得保留

- **事件相机感知领域的重要系统论文**：AMI-EV 通过硬件-软件协同设计解决事件相机内禀缺陷，项目页提供的多场景视频对比是该方法优势的直观证据。
- **工具链价值**：开源仿真器与 Translator 使研究者无需物理硬件即可开发和测试 AMI-EV 相关算法；降低事件相机研究门槛。
- **与 KEMO 等事件相机应用工作的上游关系**：AMI-EV 解决感知层问题，与高层任务（如 KEMO 中事件驱动的 VLA）形成感知–决策链路。

## 关联资料

- 论文归档：[`sources/papers/microsaccade_inspired_event_camera_scirobotics_2024.md`](../papers/microsaccade_inspired_event_camera_scirobotics_2024.md)
- Wiki 实体：[`wiki/entities/paper-microsaccade-inspired-event-camera.md`](../../wiki/entities/paper-microsaccade-inspired-event-camera.md)

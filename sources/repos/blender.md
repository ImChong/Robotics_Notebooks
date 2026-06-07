# Blender（官方源码仓库）

- **标题**：Blender
- **类型**：repo
- **来源**：Blender Foundation
- **链接**：<https://projects.blender.org/blender/blender>
- **克隆**：`https://projects.blender.org/blender/blender.git` / `git@git.blender.org:blender/blender.git`
- **入库日期**：2026-06-07
- **一句话说明：** Blender Foundation 维护的官方 **3D 创作套件** 源码仓库（C/C++ 为主），涵盖建模、绑定、动画、模拟、渲染、合成与视频编辑全管线；整体以 **GNU GPL v3** 许可发布。
- **沉淀到 wiki：** 是 → [`wiki/entities/blender.md`](../../wiki/entities/blender.md)

## 仓库概况（2026-06-07 API / README）

| 字段 | 值 |
|------|-----|
| 托管 | [projects.blender.org](https://projects.blender.org)（Blender 自建 Forgejo/Gitea 实例） |
| 默认分支 | `main` |
| 主要语言 | C++ |
| 描述 | The official Blender project repository. |
| Issue / PR | 公开 issue 与 pull request 工作流 |

## README 摘要

> Blender is the free and open source 3D creation suite. It supports the entirety of the 3D pipeline—modeling, rigging, animation, simulation, rendering, compositing, motion tracking and video editing.

**开发与文档入口（README 链接）：**

- 构建手册：<https://developer.blender.org/docs/handbook/building_blender/>
- Code Review & Bug Tracker：<https://projects.blender.org>
- 开发者论坛：<https://devtalk.blender.org>
- 开发者文档：<https://developer.blender.org/docs/>
- 用户手册：<https://docs.blender.org/manual/en/latest/index.html>

**许可：** 项目整体 **GPL-3.0**；个别文件可能采用兼容的其它许可。详见 <https://www.blender.org/about/license>。

## 与机器人研究/工程的关联点

- **资产与场景**：网格/材质/灯光/相机导出，供 Omniverse/Isaac、自定义仿真或 NeRF/3DGS 管线消费；室内场景生成论文（如 HomeWorld）中常见 **Blender shell** 作显式 3D 约束。
- **动画与动捕后处理**：**BVH**、骨骼绑定、曲线编辑是 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) 链路上游；[SAM3DBody-cpp](./sam3dbody-cpp.md) 等提供 Blender 插件导出 BVH。
- **插件宿主**：离线物理/渲染求解器（如 [ppf-contact-solver](./ppf-contact-solver.md)）通过 **Blender 5+ 远程插件** 把 GPU 仿真接到 DCC 工作流。
- **脚本化**：Python API 可批量处理网格、相机轨迹与渲染，适合数据集合成与可视化 pipeline。

## 对 wiki 的映射

- 升格页面：[wiki/entities/blender.md](../../wiki/entities/blender.md)
- 交叉引用：[wiki/entities/nvidia-omniverse.md](../../wiki/entities/nvidia-omniverse.md)、[wiki/concepts/character-animation-vs-robotics.md](../../wiki/concepts/character-animation-vs-robotics.md)、[wiki/entities/robot-motion-keyframe-editors.md](../../wiki/entities/robot-motion-keyframe-editors.md)

## 参考链接

- 源码仓库：<https://projects.blender.org/blender/blender>
- 官网：<https://www.blender.org/>
- 开发者中心：<https://developer.blender.org/>

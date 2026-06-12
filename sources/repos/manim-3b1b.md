# ManimGL / 3Blue1Brown 原版 Manim（3b1b/manim）

- **标题**：ManimGL（Grant Sanderson 原版）
- **类型**：repo
- **来源**：Grant Sanderson（3Blue1Brown 作者）
- **链接**：<https://github.com/3b1b/manim>
- **克隆**：`https://github.com/3b1b/manim.git`
- **入库日期**：2026-06-12
- **一句话说明：** 3Blue1Brown 作者为制作教育数学视频而开源的 **Python 动画引擎**（MIT）；PyPI 包名为 **`manimgl`**（非 `manim`），CLI 为 `manimgl`，以 **OpenGL** 实时预览见长，与社区版 [ManimCE](./manim-community.md) **API 与安装互不兼容**。
- **沉淀到 wiki：** 是 → [`wiki/entities/manim.md`](../../wiki/entities/manim.md)（与 ManimCE 合并叙述）

## 仓库概况（2026-06-12 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub |
| 默认分支 | `master` |
| 主要语言 | Python |
| Stars | ~87k |
| 描述 | Animation engine for explanatory math videos |
| PyPI 包名 | `manimgl` |
| CLI | `manimgl` / `manim-render` |
| 许可 | MIT |
| 文档站 | <https://3b1b.github.io/manim/> |

## README 摘要

> Manim is an engine for precise programmatic animations, designed for creating explanatory math videos.

**两版本关系（README 明示）：**

- 本仓库为 Grant 个人项目起点；2020 年社区分叉出 [ManimCommunity/manim](./manim-community.md)。
- Grant **仍维护本仓库**；视频场景专用代码在 [3b1b/videos](https://github.com/3b1b/videos)。
- **切勿混装**：用 ManimCE 的安装说明装本仓库（或反之）会导致依赖冲突；须先选定版本再按对应 README 操作。

**依赖（README）：**

- Python 3.7+
- **FFmpeg**、**OpenGL**（必需）
- **LaTeX**（可选，用于公式排版）
- Linux 另需 **Pango** 开发头文件（文本渲染）

**最小示例：**

```sh
pip install manimgl
manimgl example_scenes.py OpeningManimExample
```

## 与 ManimCE 的关键差异（选型摘要）

| 维度 | ManimGL（本仓库） | ManimCE（社区版） |
|------|-------------------|-------------------|
| 包名 / CLI | `manimgl` / `manimgl` | `manim` / `manim` |
| 维护主体 | Grant Sanderson | Manim Community |
| 文档 | 3b1b.github.io/manim | docs.manim.community |
| 社区活跃度 | 以作者节奏为主 | issue/PR、插件、多语言文档更活跃 |
| 典型用途 | 复现 3B1B 制作手法、OpenGL 交互预览 | 新用户入门、插件生态、长期项目维护 |

## 与机器人研究/工程的关联点

- **溯源 3Blue1Brown 风格讲解**：控制/优化/几何类公开课程常借鉴其 **「程序化 + 精确」** 叙事；本仓库是理解该风格的 **一手代码基线**。
- **非仿真工具**：与 [MuJoCo](../../wiki/entities/mujoco.md) / [Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) 无交集；仅服务 **对外解释层**。
- **视频资产管线**：输出 MP4（经 FFmpeg）；可嵌入论文主页、课程平台或社交媒体，与 [Blender](../../wiki/entities/blender.md) 渲染的 3D 演示片互补。

## 对 wiki 的映射

- 升格页面：[wiki/entities/manim.md](../../wiki/entities/manim.md)
- 对照仓库：[sources/repos/manim-community.md](./manim-community.md)

## 参考链接

- 仓库：<https://github.com/3b1b/manim>
- 文档：<https://3b1b.github.io/manim/>
- 3Blue1Brown 官网：<https://www.3blue1brown.com/>
- 视频专用代码：<https://github.com/3b1b/videos>

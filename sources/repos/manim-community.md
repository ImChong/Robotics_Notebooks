# Manim Community Edition（ManimCommunity/manim）

- **标题**：Manim Community Edition（ManimCE）
- **类型**：repo
- **来源**：Manim Community（社区维护）
- **链接**：<https://github.com/ManimCommunity/manim>
- **克隆**：`https://github.com/ManimCommunity/manim.git`
- **入库日期**：2026-06-12
- **一句话说明：** 社区维护的 **Python 数学动画引擎**（MIT），从 [3b1b/manim](./manim-3b1b.md) 分叉而来；以 `pip install manim` 安装、`manim` CLI 渲染，配套完整文档、插件生态与活跃 issue/PR 工作流。
- **沉淀到 wiki：** 是 → [`wiki/entities/manim.md`](../../wiki/entities/manim.md)

## 仓库概况（2026-06-12 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub |
| 默认分支 | `main` |
| 主要语言 | Python |
| Stars | ~39k |
| 描述 | A community-maintained Python framework for creating mathematical animations. |
| PyPI 包名 | `manim` |
| CLI | `manim` |
| 许可 | MIT |

## README 摘要

> Manim is an animation engine for explanatory math videos. It's used to create precise animations programmatically, as demonstrated in the videos of 3Blue1Brown.

**与原版关系（README 明示）：**

- ManimCE 由社区从 Grant Sanderson 的 [3b1b/manim](./manim-3b1b.md) **分叉**并持续开发。
- 官方 README **推荐新用户安装 ManimCE**（文档更全、社区更活跃、功能迭代更快）。
- 若目标是研究 Grant 本人如何制作 3Blue1Brown 视频，应去看原版仓库与其 [3b1b/videos](https://github.com/3b1b/videos) 视频专用代码。

**安装与运行（README 要点）：**

```sh
pip install manim
manim -p -ql example.py SquareToCircle
```

- 在线试用：<https://try.manim.community/>（Jupyter）
- Docker 镜像：`manimcommunity/manim`
- 文档：<https://docs.manim.community/en/stable/>

## 核心架构概念（文档 v0.20.1 摘要）

| 概念 | 说明 |
|------|------|
| **Scene** | 动画场景容器；子类实现 `construct()`，用 `self.play()` 编排时间线 |
| **Mobject** | 数学对象（`Circle`、`MathTex`、`NumberPlane`、`Graph` 等）的可组合基类 |
| **Animation** | `Create`、`Transform`、`FadeIn`、`Write` 等时间演化算子 |
| **Camera** | 2D/3D/移动/缩放相机场景变体（`ThreeDScene`、`MovingCameraScene` 等） |
| **Renderer** | 支持 Cairo / OpenGL 等后端（见安装 FAQ 与配置文档） |
| **Plugins** | 独立 Python 包扩展核心库（<https://docs.manim.community/en/stable/plugins.html>） |

## 与机器人研究/工程的关联点

- **算法与理论可视化**：用 `MathTex` / `NumberPlane` / `Graph` 把 RL 回报、李群几何、优化轨迹等 **精确程序化** 成讲解短片——适合论文 supplementary、课程与组会，而非替代仿真器。
- **与 DCC 工具对照**：[Blender](../../wiki/entities/blender.md) 管 **3D 资产与角色动画**；Manim 管 **2D/轻 3D 数学示意**——机器人栈里二者常 **并列** 出现在「对外沟通层」。
- **教学演示**：与 [BotLab / MotionCanvas](../../wiki/entities/botlab-motioncanvas.md) 等 **交互式浏览器仿真** 互补：Manim 产出 **离线、可复现的讲解视频**；BotLab 侧重 **在线调参**。
- **许可友好**：MIT 许可，科研组可自由修改与分发生成视频（注意 3Blue1Brown 的 **Pi creature** 等版权角色不可挪用，见官方 License 说明）。

## 对 wiki 的映射

- 升格页面：[wiki/entities/manim.md](../../wiki/entities/manim.md)
- 交叉引用：[wiki/entities/blender.md](../../wiki/entities/blender.md)、[wiki/entities/botlab-motioncanvas.md](../../wiki/entities/botlab-motioncanvas.md)、[wiki/concepts/character-animation-vs-robotics.md](../../wiki/concepts/character-animation-vs-robotics.md)

## 参考链接

- 仓库：<https://github.com/ManimCommunity/manim>
- 社区站：<https://www.manim.community/>
- 稳定版文档：<https://docs.manim.community/en/stable/>
- 安装 FAQ（两版本区别）：<https://docs.manim.community/en/stable/faq/installation.html>

# Manim Community 官网与文档（manim.community）

- **类型**：网站 / 产品主页 + 官方文档
- **入口**：
  - 社区主页：<https://www.manim.community/>
  - 稳定版文档：<https://docs.manim.community/en/stable/>
  - 在线 playground：<https://try.manim.community/>
- **主体**：Manim Community（非营利社区组织，MIT 许可软件）
- **收录日期**：2026-06-12
- **抓取说明**：以 **2026-06-12** 对首页与文档索引的公开文案为准；当前文档标注版本 **Manim Community v0.20.1**。

## 一句话

**Manim Community Edition（ManimCE）** 是由社区维护的 **Python 数学动画库**，用代码精确描述每一帧如何演化，以降低「技术概念动画化」的手工成本；官网承担产品定位、示例画廊、赞助与贡献入口，文档站提供安装、教程、API 参考与插件指南。

## 为什么值得保留

- 机器人知识库大量涉及 **RL、控制、几何、优化** 等需要对外讲解的主题，但此前缺少 **程序化数学动画** 工具的独立溯源节点。
- 与 [Blender](../repos/blender.md)（3D DCC）、[BotLab MotionCanvas](../../wiki/entities/botlab-motioncanvas.md)（浏览器交互仿真）形成 **对外沟通工具链** 的对照：Manim 擅长 **离线、可版本控制的公式/示意图视频**。
- 官网与文档 **明确区分** ManimCE 与 [3b1b/manim](../repos/manim-3b1b.md) 两个不兼容版本，避免维护者混装踩坑。

## 官网公开要点（2026-06-12）

| 模块 | 内容 |
|------|------|
| **定位** | A community maintained Python library for creating mathematical animations |
| **起源** | 最初由 Grant Sanderson（3Blue1Brown）编写并开源；现由 Manim Community 维护 |
| **许可** | MIT；生成视频可自由分享，不强制署名 Manim |
| **示例画廊** | 首页嵌入 `Scene` 代码与渲染结果（`MathTex`、`NumberPlane`、`StreamLines` 等） |
| **生态入口** | Plugins、Getting Started、Contribute、Translate、Discord / Reddit |
| **赞助** | 列出支持组织（首页 Sponsors 区块） |

## 文档站结构摘要（stable，v0.20.1）

| 分区 | 用途 |
|------|------|
| **Example Gallery** | 按主题分类的代码+成片对照 |
| **Installation** | pip / Conda / Docker / Jupyter；强调与 ManimGL 不可混装 |
| **Tutorials & Guides** | Quickstart、输出设置、building blocks、配置、文本与公式、旁白等 |
| **Reference Manual** | `Scene`、`Mobject`、`Animation`、相机、配置等完整 API |
| **Plugins** | 第三方扩展的安装与开发 |
| **Changelog** | 版本变更记录（当前页眉 v0.20.1） |
| **Contributing** | 开发流程、文档、测试、国际化 |

**文档特别强调的许可注意：**

- Grant Sanderson 视频中的 **Pi creatures** 等角色资产 **受版权保护**，衍生作品应避免直接使用。
- 用 Manim 制作的视频可自由发布；学术引用见 GitHub README 的 cite 说明。

## 对 wiki 的映射

- 升格页面：[wiki/entities/manim.md](../../wiki/entities/manim.md)
- 代码仓库：[sources/repos/manim-community.md](../repos/manim-community.md)
- 原版对照：[sources/repos/manim-3b1b.md](../repos/manim-3b1b.md)

## 参考链接

- 官网：<https://www.manim.community/>
- 文档：<https://docs.manim.community/en/stable/>
- 在线试用：<https://try.manim.community/>
- GitHub：<https://github.com/ManimCommunity/manim>
- 两版本 FAQ：<https://docs.manim.community/en/stable/faq/installation.html>

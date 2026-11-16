# ATLAS 官方网站源码：atlas-humanoid.github.io

> 来源归档（repo / 开放边界核验）

- **标题：** atlas-humanoid/atlas-humanoid.github.io
- **类型：** repo（ATLAS 项目网站与演示素材生成工具；非算法训练代码）
- **URL：** <https://github.com/atlas-humanoid/atlas-humanoid.github.io>
- **项目页：** <https://atlas-humanoid.github.io/>（[项目页归档](../sites/atlas-humanoid.md)）
- **技术博客：** <https://atlas-humanoid.github.io/blog/>（[官方技术原文精读](../blogs/atlas_humanoid_early_release.md)）
- **核查日期：** 2026-11-14
- **核对范围：** GitHub `main` 递归文件树（72 个文件）、`tools/learning-loop/README.md`、`tools/opening-showcase/README.md`；检索训练、部署与数据入口；未逐行审阅全部网站 JS/TS/Python。
- **通读：** 未完成
- **一句话说明：** 开放的是 ATLAS 网站的 HTML/CSS/JS、视频、示意动画制作工具；没有可验证的 ATLAS 在线强化学习、真机跟踪控制或部署入口。

## 仓库里实际有什么

| 路径 | 可核实的作用 | 是否是 ATLAS 策略实现 |
|---|---|---|
| `index.html`, `blog/index.html` | 官方主页与 Early release 技术博客页面 | 否 |
| `css/`, `js/` | 网站样式与页面交互脚本 | 否 |
| `media/` | 展示视频、海报与图片素材 | 否 |
| `tools/learning-loop/` | Remotion/TSX 视频生成：将真实片段与**概念性**学习回路动画拼合；README 明确说明动画不代表实测轨迹 | 否 |
| `tools/opening-showcase/render.py` | 利用 FFmpeg/FFprobe 合成首页视频与海报并验证视频属性 | 否 |

网站的 `tools/learning-loop/README.md` 提供 `npm ci`、`npm run render` 等**视频渲染**命令；`tools/opening-showcase/README.md` 提供 `python3 tools/opening-showcase/render.py` **视频导出**命令。**这些是宣传/可视化素材生成流程，不是 BPTT 在线训练命令，不能当成可部署的控制器。**

## 与 SuperMimic 的边界

ATLAS [技术博客 §2.3](https://atlas-humanoid.github.io/blog/) 描述利用真实机器人抓取尝试拟合末端命令–响应动力学，再通过模型 rollout / BPTT 更新残差 actor。这在方法层面与 [SuperMimic](https://supermimic.github.io/) 的可微动力学优化路线有关，但 ATLAS 的 **System 0 是冻结 ScaleBFM**，而非从本仓库获取的 SuperMimic 跟踪策略。

**开源判断（2026-11-14）：** 网站及展示工具代码**已公开**；ATLAS 的模型训练、在线 actor 更新、真机控制通讯/部署程序、预训练权重与完整实验资产**未在该仓库公开**。未检查到可供 ATLAS 算法复现的运行入口，故对应知识节点的**源码运行时序图不适用**。不能把网站开源等同于论文方法开源。

## 对 wiki 的映射

- [ATLAS：Agentic Harness](../../wiki/entities/atlas-agentic-harness.md) — 唯一项目实体节点，分清 ScaleBFM 基座、SuperMimic 方法关联与网站代码边界。
- [ATLAS 官方技术博客归档](../blogs/atlas_humanoid_early_release.md) — 机制、实验、局限以原文为准。

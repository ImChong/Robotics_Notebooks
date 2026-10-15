# Get It. 项目页（getit.noesisai.it）

> 来源归档

- **标题：** Get It. — Read it. See it. Get it.
- **类型：** site（开源 AI 学习桌面应用官网与下载入口）
- **URL：** <https://getit.noesisai.it/>
- **代码：** <https://github.com/beltromatti/get-it>（公开，Apache-2.0）
- **入库日期：** 2026-10-15
- **一句话说明：** 展示围绕 PDF 的概念可视化、知识图谱、四类复习工具和掌握度评估，并提供多平台桌面版下载入口。
- **代码状态：** 已开源；项目页提供 GitHub source 链接。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Problem | 目标从摘要/卡片记忆转向检验学习者能否回答陌生问题 |
| Visualizer | 页面概念标签与 Three.js、Canvas、公式、曲线和参考资料等 renderer |
| Knowledge graph | 文档概念图，每个节点显示四个维度的分数及关系 |
| Four tools | Chat、flashcards、quizzes、Feynman 四类交互汇入同一学习日志 |
| Evaluator | agent 读取学习记录，展示记忆、理解、结构与应用分数 |
| Install | 多平台桌面应用下载、版本更新和社区入口 |

## 开源与版本说明

- 项目页标注 **Open source · Apache 2.0 · v1.1.0**。
- 页面仍称下载包是未签名的 hackathon build；GitHub 仓库 README 对较新 v1.2.1 版本说明 macOS 构建已签名并通过 notarization。此处保留两份官方资料的差异，实际安装前以对应 Release 产物与说明为准。
- 页面声明文档以本地应用为中心，模型通过用户选择的官方 CLI/API 访问供应方；模型数据流、供应商条款与应用本地存储应分别理解。

## 对 wiki 的映射

- 主实体：[Get It. Study Companion](../../wiki/entities/get-it-study-companion.md)
- 代码仓库：[beltromatti-get-it.md](../repos/beltromatti-get-it.md)

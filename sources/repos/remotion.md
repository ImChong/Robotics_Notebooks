# Remotion

> 来源归档

- **标题：** Remotion
- **类型：** repo
- **链接：** https://github.com/remotion-dev/remotion
- **Stars：** ~61k+（2026-09-30 API）
- **许可：** **Remotion License**（非 SPDX 单一 OSI 许可；公司与大规模使用见 [remotion.dev/license](https://www.remotion.dev/license)）
- **入库日期：** 2026-09-30
- **一句话说明：** 用 **React 组件作视频源真值** 的可编程视频框架：代理时代强调 agent 生成/编辑视频、交互式时间轴、批量渲染（Node/Lambda/Vercel/客户端）与 Player/Editor 嵌入。
- **为什么值得保留：** 与本站 [video-shotcraft](../../wiki/entities/video-shotcraft.md)、[motion-control 路线 Remotion 成片](../../media/roadmap-motion-control-video/README.md) 及 agent 内容生产栈直接相关。
- **沉淀到 wiki：** 是 → [`wiki/entities/remotion.md`](../../wiki/entities/remotion.md)
- **代码：** **已开源**（GitHub monorepo；商业许可条款见官网）

---

## 定位（README 2026）

- **Video tools for the agent era** — agentic / interactive / programmatic 三条路径可互转。  
- React 代码 = source of truth；设计系统、百万级 batch、SaaS 视频编辑器场景。  
- 文档 >1000 页；含 [Agent Skills](https://www.remotion.dev/docs/ai/skills)、[Prompts](https://www.remotion.dev/prompts)、模板与 `@remotion/*` 生态。

## 快速开始

```bash
npx create-video@latest
```

## 渲染面

| 模式 | 用途 |
|------|------|
| Node SSR | 本地/服务器渲染 |
| Lambda | 云批量 |
| Vercel Sandbox | 托管渲染 |
| Client-side | 浏览器内导出 |
| Player / Editor Starter | 应用内预览与编辑 |

## 与本站关系

- [video-shotcraft](video-shotcraft.md) 技能库内置 161 条 Remotion 动态样片与 `npx remotion still` 验收。  
- 机器人知识库 **roadmap 解说视频** 可选用 Remotion + 章节脚本 pipeline（见 `media/roadmap-motion-control-video/`）。

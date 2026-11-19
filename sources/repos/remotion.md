# Remotion

> 来源归档

- **标题：** Remotion
- **类型：** repo
- **链接：** https://github.com/remotion-dev/remotion
- **Stars：** ~61k+（2026-09-30 API）
- **许可：** **Remotion License**（非 SPDX 单一 OSI 许可；公司与大规模使用见 [remotion.dev/license](https://www.remotion.dev/license)）
- **入库日期：** 2026-09-30
- **复核日期：** 2026-11-19
- **核对范围：** 官方 GitHub README（Agentic / Interactive / Programmatic、渲染面、安装、文档）、`packages/cli/README.md`、`packages/cli/src/index.ts`（CLI 命令分发）、`packages/cli/src/render-flows/render.ts`（浏览器及 render pipeline）、`LICENSE.md`（免费/公司许可）。
- **通读：** 已完成
- **一句话说明：** 用 **React 组件作视频源真值** 的可编程视频框架：代理时代强调 agent 生成/编辑视频、交互式时间轴、批量渲染（Node/Lambda/Vercel/客户端）与 Player/Editor 嵌入。
- **为什么值得保留：** 与本站 [video-shotcraft](../../wiki/entities/video-shotcraft.md)、[motion-control 路线长视频成片（Playwright + FFmpeg 管线，与 Remotion 同属代码驱动视频路线对照）](../../media/roadmap-motion-control-video/README.md) 及 agent 内容生产栈直接相关。
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


## 2026-11-19 一手资料复核

- **运行入口：** `npx create-video@latest` 初始化；项目内 `npx remotion studio` 预览、`npx remotion render` 出片、`npx remotion still` 出单帧。CLI 的 `packages/cli/src/index.ts` 将 `render` 分派至 `./render`，渲染流 `render-flows/render.ts` 处理浏览器启动、打包、逐帧绘制和编码过程。无需把 Studio 当成批处理必经步骤。
- **许可精确边界：** 仓库 `LICENSE.md` 的免费许可允许个人、**最多 3 名员工的营利组织**、非营利组织及非商业评估；不符合这些条件的组织需 Company License。Remotion 5.0 许可可能调整，商业使用须重新查当时官方条款。
- **官方路径：** [README](https://github.com/remotion-dev/remotion/blob/main/README.md) · [CLI README](https://github.com/remotion-dev/remotion/blob/main/packages/cli/README.md) · [CLI 入口](https://github.com/remotion-dev/remotion/blob/main/packages/cli/src/index.ts) · [render 流程](https://github.com/remotion-dev/remotion/blob/main/packages/cli/src/render-flows/render.ts) · [许可全文](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md) · [首个视频教程](https://www.remotion.dev/docs/your-first-video)。
- **与 HyperFrames 比较：** Remotion 以 React Component 为视频源；HyperFrames 以可直接播放的 HTML/CSS 为源，二者均用于**视频生产**，不涉及机器人控制执行器本体。

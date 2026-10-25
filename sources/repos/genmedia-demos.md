---
type: source
source_type: repo
title: "GenMedia Demos — Elsewhere, Worldline, FLUX 3 Action"
url: "https://github.com/lovisdotio/genmedia-demos"
project_id: genmedia-demos
code: "https://github.com/lovisdotio/genmedia-demos"
status: public
updated: 2026-10-25
summary: "Lovis Odin 在 GenMedia Conference 2026 展示的三个基于 fal 的生成媒体交互演示：分支故事、配送机器人 Sim2Real 可视化、SO-101 动作回放。"
---

# GenMedia Demos（Lovis Odin）

> 来源归档：三个交互演示的开源实现，重点关注生成媒体如何辅助机器人仿真可视化与动作策略呈现。

- **官方代码：** https://github.com/lovisdotio/genmedia-demos
- **作者发布：** https://www.linkedin.com/posts/lovis-odin-7a751360_worldline-elsewhere-flux-3-action-three-activity-7511883088726052865-0sLX
- **技术栈：** Next.js / 3D 场景 / fal 生成媒体接口；网站回放预生成结果，浏览时不调用生成 API。
- **本地运行：** Node.js 22+，执行 `npm install`、`npm run dev`；`npm run build` 静态导出。

## 三个演示

| 演示 | 入口 | 核心内容 |
| --- | --- | --- |
| Elsewhere | [stories](https://github.com/lovisdotio/genmedia-demos/tree/main/app/stories) | 183 个生成场景、76 条记录路径，组成可连续播放或逐帧拖动的 3D 分支故事树。 |
| Worldline · Delivery robot | [sim2real](https://github.com/lovisdotio/genmedia-demos/tree/main/app/sim2real) | 同一模拟街道的 12 种未来分支，3D 预演转写实视频，并以 SAM 3 对照仿真；包含白天和雪天。 |
| FLUX 3 Action · Robot arm | [flux3](https://github.com/lovisdotio/genmedia-demos/tree/main/app/flux3) | SO-101 策略与 LeRobot 真实录制对比；在 3D 机械臂回放动作并重新渲染视频。 |

## 上游模型和生成流水线

- `minimax/h3-max/director`：Elsewhere 场景生成。
- `minimax/h3-max/reference-to-video`：配送机器人与机械臂的 3D 预演转视频。
- `fal-ai/nano-banana-2/edit`：仿真首帧的写实化（白天、雪天）。
- `fal-ai/sam-3/video-rle`：生成视频与仿真一致性检查。
- `fal-ai/flux-3-action/so101`：SO-101 动作生成。
- `fal-ai/bria/background/remove`：Elsewhere 人物抠图。

## 机器人研究价值与边界

1. **Sim2Real 展示：** 将同一仿真轨迹转换成不同天气和外观的视频，便于展示、检查视觉一致性；这并不等同于已验证的真机迁移。
2. **动作策略可解释展示：** 结合 SO-101 关节动作、3D 机器人和生成视频，让动作序列可回放、可对照。
3. **区分 demo 与模型：** 此仓库主要是交互体验与预计算媒体回放；FLUX 3 Action 的训练和推理实现应参考上游独立仓库。

## 一手来源

- [Lovis Odin / GenMedia Demos 源码及 README](https://github.com/lovisdotio/genmedia-demos)
- [作者公开发布](https://www.linkedin.com/posts/lovis-odin-7a751360_worldline-elsewhere-flux-3-action-three-activity-7511883088726052865-0sLX)
- [Black Forest Labs / FLUX 3 Action 源码](https://github.com/black-forest-labs/flux-action)
- [FLUX 3 Action SO-101 权重与使用说明](https://huggingface.co/black-forest-labs/flux-3-action-so101)

## 关联知识

- [FLUX 3 Action](../../wiki/entities/flux-3-action.md)
- [World Action Models](../../wiki/concepts/world-action-models.md)
- [Sim2Real](../../roadmap/depth-sim2real.md)

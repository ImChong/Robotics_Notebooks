# Particles4All（matsuoka-601/Particles4All）

> 来源归档（repo）

- **标题：** Particles4All
- **类型：** repo
- **作者：** matsuoka-601
- **链接：** https://github.com/matsuoka-601/Particles4All
- **在线演示：** https://particles4all.netlify.app/
- **许可证：** （仓库未在 API 返回 SPDX；以 GitHub 页面为准）
- **入库日期：** 2026-09-13
- **一句话说明：** 浏览器 **WebGPU** 实时统一粒子物理：流体与刚体同为粒子，在同一 **PBD** 约束环中求解；含浮力、表面张力、各向异性核流体表面重建与 screen-space narrow-range 流体渲染。
- **沉淀到 wiki：** 是 → [`wiki/entities/particles4all.md`](../../wiki/entities/particles4all.md)

---

## 开源状态（步骤 2.5，截至 2026-09-13）

| 资源 | 状态 |
|------|------|
| 源码 | **已开源** — ES modules + WGSL，无构建步骤 |
| 在线 Demo | https://particles4all.netlify.app/ |
| 依赖 | 仅需本地 `python -m http.server`（WebGPU 需安全上下文） |

**结论：确认已开源，浏览器即跑。**

## 技术要点（README 核对）

- **统一求解：** 流体与刚体均离散为粒子，单套 **PBD constraint solver**（参考 NVIDIA Flex 统一粒子物理思路）。
- **浮力：** 在统一求解器内自然出现。
- **表面张力：** 基于 SPH 表面张力/黏附文献实现。
- **渲染：** 各向异性核重建流体表面 + screen-space **narrow-range filter** 平滑。
- **交互：** 左键旋转、右键平移、滚轮缩放、悬停推水、空格暂停；**H** 隐藏 UI、**D** 调试窗。

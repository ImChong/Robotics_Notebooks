# Lucida 项目页（lucida-r2s.github.io）

> 来源归档（ingest 附属）

- **标题：** Lucida: Parse, Generate, and Place for Composable Real-to-Sim Scene Modeling — Project Page
- **类型：** site / project page
- **链接：** <https://lucida-r2s.github.io/>
- **关联论文：** [lucida_r2s_arxiv_2608_30821.md](../papers/lucida_r2s_arxiv_2608_30821.md)（arXiv:2608.30821）
- **机构：** 字节跳动 Seed（ByteDance Seed）；北京大学（PKU）；浙江大学（ZJU）
- **入库日期：** 2026-09-04
- **代码核查（2026-09-04）：** **确认未开源** — `project.resources` 仅挂 **arXiv** 与 **Hugging Face papers**；无 GitHub / 权重 / 数据集链接。论文正文只列项目页，未承诺 code will be released。`github.com/lucida-r2s` 为 Pages 托管入口，公开检索无实现仓。
- **一句话说明：** 室内多视角捕获 → 可编辑物体资产 + 场景图的 Real-to-Sim 项目页：Parse / Generate / Place（GizmoAct）管线、交互 3D 场景、位姿与场景重建数字。

## 页面结构（2026-09-04 抓取）

1. **Hero / Resources** — 标题、作者、机构；资源按钮由 `assets/js/project-data.js` 渲染：`arXiv` → <https://arxiv.org/abs/2608.30821>，`Hugging Face` → <https://huggingface.co/papers/2608.30821>，另有页内 BibTeX。
2. **Overview** — 把真实室内场景重建为完整、可编辑的物体资产并按观测布置；精度留到闭环末段。含 teaser 视频。
3. **Method** — 三阶段：Parse（关键帧 + 实例关联 + 带证据的场景图）→ Generate（无遮挡物体图 + 可编辑 3D 资产）→ Place（粗初始化后 GizmoAct 闭环位姿编辑）。
4. **Demo** — 物体级可编辑 mesh；PlayCanvas 交互场景（`playcanv.as/e/p/qd51UfXt/`）。
5. **GizmoAct** — VLM 操作 3D 编辑器：渲染观测 → 物体局部系可执行编辑 → 自行决定停止；含 easy/medium/hard 初始化画廊与 Boxer / Any6D* / SAM 3D 鲁棒性对照。
6. **Results** — 场景级 3D 检测（Boxer / WildDet3D）、物体位姿（R2S-Object / CA-1M / ADT）、R2S-Scene 场景重建；定性图可展开。
7. **Acknowledgements / BibTeX** — `@article{lucida2026, … eprint={2608.30821}}`。页脚标 “Project page draft”。

## 对 wiki 的映射

- 实体页：[paper-lucida-r2s.md](../../wiki/entities/paper-lucida-r2s.md)
- 概念对照：[Sim2Real](../../wiki/concepts/sim2real.md) Real2Sim 节

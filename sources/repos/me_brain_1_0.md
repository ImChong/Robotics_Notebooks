# ME-Brain-1.0 官方仓库

> 来源归档（repo）

- **标题：** ME-Brain-1.0: Memory, Cognition, and Action for Self-Evolving Embodied Intelligence
- **类型：** repo
- **链接：** https://github.com/MachEmbodied/ME-Brain-1.0
- **arXiv：** <https://arxiv.org/abs/2609.24271>
- **关联动作模型：** https://github.com/MachEmbodied/Focus-VLWA（核对版本 `main@0aaf1f5`，HF 权重 `zhengjingxie/focus-vlwa`）
- **初次入库：** 2026-09-25
- **仓库复核日期：** 2026-11-21
- **核对范围：** 阅读 `MachEmbodied/ME-Brain-1.0` `README.md` 与关联动作模型 `MachEmbodied/Focus-VLWA@0aaf1f5` 的 `README.md`、`pyproject.toml`；核实 `ME-Brain-1.0` 主仓的 `me_brain/`/`robot/`/`simulation/`/`ME-VLM` 仍为 Todo 占位，而 `Focus-VLWA` 已发布训练/推理代码（`focus-vlwa post-train`、`FocusVLWAPolicy`、`HeadHistoryBuffer`）及 HF 权重 `zhengjingxie/focus-vlwa`；未在本地 GPU 运行微调或评测。
- **通读：** 已完成
- **一句话说明：** 自我演进具身框架占位仓；Focus-VLWA 动作模型训练/推理已释（README News 日期 2026-09-22；2026-10-09 复核）。
- **沉淀到 wiki：** [`wiki/entities/paper-me-brain-1-0.md`](../../wiki/entities/paper-me-brain-1-0.md)

## 开源状态

- **部分开源（2026-10-09 复核）：** README 的 Repository Structure 列出 `me_brain/`、`robot/`、`simulation/`，并将 Focus-VLWA 与 ME-VLM 作为子模块；但 Todo 仍将 ME-VLM、完整 ME-Brain Framework、Simulation integration、Real Robot integration 标为未完成。
- **已发布：** Focus-VLWA 训练与推理代码（News 日期 2026-09-22）。**待发布：** Focus-VLWA 预训练权重、Cognition Core/ME-VLM、完整框架及仿真/真机集成。
- **许可：** README 声明 Apache-2.0；第三方依赖、子模块、权重和数据集分别遵循各自许可。

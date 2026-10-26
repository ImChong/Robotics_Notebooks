# ARROW — 官方代码仓库

> 来源归档（ingest · GitHub 仓库）

- **类型：** repo / research-code
- **官方仓库：** <https://github.com/ifradlin/arrow>
- **论文：** <https://arxiv.org/abs/2610.01314>
- **项目页：** <https://ifradlin.github.io/arrow/>
- **权重：** <https://huggingface.co/ifradlin/arrow>
- **机构：** 亚琛工业大学（RWTH Aachen University）
- **开源状态：** 已开放推理代码、预训练权重与交互式 3D 查看器
- **许可：** ARROW 自有代码与查看器 MIT；部分改编模型源码 Apache-2.0；预训练权重 CC BY-NC 4.0，且受 Waymo 非商业条款约束
- **入库日期：** 2026-10-26
- **一句话说明：** ARROW 官方推理实现，支持从图像集、视频和多视角输入执行 3D 重建、点跟踪及相机相关任务，并可在浏览器查看结果。

## README 要点（编译自上游）

- **安装：** `uv sync`；安装交互查看器依赖：`uv sync --extra viewer`。
- **浏览器查看器：** `uv run arrow-viewer`，默认在 `http://127.0.0.1:7860` 提供界面；可直接选择图像或视频运行推理。
- **支持任务：** 3D tracking、3D reconstruction、camera poses、video depth、all-pixel tracking；多视角输入支持前三类。
- **命令行推理：** `uv run arrow infer assets/videos/railway.mp4 --stride 3 --output results/railway.npz`。
- **查看已保存结果：** `uv run arrow-viewer --scene results/railway.npz`。
- **Python API：** `ARROW.from_pretrained("ifradlin/arrow")` 配合 `load_input` 与 `infer`；首次运行会下载权重。
- **许可证边界：** 权重是非商业许可；README 的 License 进一步列出对 Waymo 条款的约束，不能把代码 MIT 误读成权重也可商用。

## 为什么值得保留

- 让任意观测集合上的动态几何与点跟踪有了可运行的官方实现。
- 交互查看器把推理任务、点查询和 3D 预测结果放在同一界面，便于复现和检查。
- 代码与权重许可边界清楚，便于评估研究使用与商业集成限制。

## 对 wiki 的映射

- [ARROW 实体页](../../wiki/entities/paper-arrow-4d-reconstruction.md)
- [ARROW 项目页](../sites/arrow-project.md)
- [ARROW 论文摘录](../papers/arrow_arxiv_2610_01314.md)
- [D4RT](../../wiki/entities/paper-d4rt.md) — 查询式动态 4D 几何的相关工作

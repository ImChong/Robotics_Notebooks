# ryhara/hand_visibility_detector

> 来源归档

- **标题：** Hand Visibility Detector（逐关节手部可见性）
- **类型：** repo
- **组织 / 作者：** Ryosei Hara 等（庆应 / AIST / 欧姆龙 SINIC X / 东京大学）
- **代码：** <https://github.com/ryhara/hand_visibility_detector>
- **默认分支：** `main`
- **权重：** <https://huggingface.co/ryhara/hand-visibility-detector>（`best.pt` / `best_hamer.pt`）
- **Demo Space：** <https://huggingface.co/spaces/ryhara/hand-visibility-detector>
- **论文：** arXiv:2608.11574 — [`sources/papers/hand_visibility_detector_arxiv_2608_11574.md`](../papers/hand_visibility_detector_arxiv_2608_11574.md)
- **入库日期：** 2026-08-15
- **一句话说明：** 冻结 WiLoR / HaMeR 骨干 + 轻量 visibility head；`HandVisibilityPipeline` 可 `pip`/`uv` 安装。**已开源、可运行**（研究/非商用）。

## 开源核查（2026-08-15）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开；默认 `main`；约 52★） |
| 项目页 | **无**独立 `*.github.io`；入口即 GitHub + HF |
| License | GitHub `license` 字段为空。README 写 **research and non-commercial use only**，须同时遵守 WiLoR / WiLoR-mini / HaMeR / MANO / HInt / COCO-WholeBody / Ultralytics 上游条款 |
| 可运行入口 | **有** — `HandVisibilityPipeline.predict`；`demo.py` / `demo_video.py` / `demo_gradio.py`；`python -m training.train` / `training.evaluate` |
| 权重 | HF `ryhara/hand-visibility-detector`：`best.pt`（WiLoR）与 `best_hamer.pt`。其余骨干（resnet / vit / cspnext / dinov2 / dinov3）无发布 ckpt，需自训后传 `vis_checkpoint`。HaMeR 骨干权重从官方 Space `geopavlakos/HaMeR` 拉 `hamer.ckpt`（约 2.5 GB，gated） |
| 数据 | 训练需自备 [HInt](https://github.com/ddshan/hint)（含 Ego4D 子集）或 COCO-WholeBody；仓内不托管全量图 |
| 结论 | **已开源**（推理包、训练、评测、发布权重、Gradio）。许可偏研究/非商用，商用前必须逐条核对上游 |

## 入口速查

| 路径 / 命令 | 作用 |
|-------------|------|
| `uv add git+https://github.com/ryhara/hand_visibility_detector.git` | 当依赖安装（也可用 pip） |
| `uv sync --extra demo` / `--extra train` | 本地跑 demo 或训练 |
| `demo.py image.jpg -o out.jpg` | 单图：WiLoR 检手 + 可见性着色 |
| `demo_video.py video.mp4` | 视频 |
| `demo_gradio.py` | 本地 Gradio |
| `src/hand_visibility_detector/pipeline.py` | `HandVisibilityPipeline` |
| `src/hand_visibility_detector/visibility_net.py` | visibility head |
| `src/hand_visibility_detector/hub.py` | HF 权重下载 |
| `python -m training.train --config training/configs/hint.yaml` | HInt、冻 WiLoR、只训 head |
| `python -m training.evaluate --config training/configs/hint_eval.yaml` | HInt 测试子集 |

**最短路径：** `uv add` 或 clone + `uv sync --extra demo` → `demo.py`（省略 `--checkpoint` 时自动下 `best.pt`）。复现论文表再备 HInt 后跑 `training.train` / `evaluate`。

README 注明：框与姿态走 **WiLoR**，逐点可见性走本文 head。

## 对 wiki 的映射

- 论文：[`sources/papers/hand_visibility_detector_arxiv_2608_11574.md`](../papers/hand_visibility_detector_arxiv_2608_11574.md)
- 沉淀 **[`wiki/entities/paper-hand-visibility-detector.md`](../../wiki/entities/paper-hand-visibility-detector.md)**

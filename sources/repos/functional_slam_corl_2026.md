# Hbelief1998/Functional-SLAM-CoRL_2026

> 来源归档

- **标题：** Functional-SLAM（CoRL 2026）
- **类型：** repo
- **组织 / 作者：** Xinggang Hu 等（清华 / 大连理工 / ETH Zurich）
- **代码：** <https://github.com/Hbelief1998/Functional-SLAM-CoRL_2026>
- **论文：** arXiv:2609.07497 — [`sources/papers/functional_slam_arxiv_2609_07497.md`](../papers/functional_slam_arxiv_2609_07497.md)
- **数据集：** <https://huggingface.co/datasets/xg-123/Functional-SLAM-dataset>
- **入库日期：** 2026-09-09
- **一句话说明：** MASt3R-SLAM 几何前端 + RAM++/DeepSeek/SAM3 开放词汇功能感知 + 在线功能图维护 + 功能拓扑回环；`main.py` 一键跑 FunGraph3D / SceneFun3D 评测序列。**已开源、可运行**。

## 开源核查（2026-09-09）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开；CoRL 2026 接收标注） |
| License | **CC BY-NC-SA 4.0**（`LICENSE`）；thirdparty（MASt3R、RAM、SAM3 等）走各自许可 |
| 可运行入口 | **有** — `python main.py --dataset ... --config config/fungraph_eval_node_place.yaml` |
| 权重 | MASt3R（Naver Labs wget）、RAM++（`xinyu1205/recognize-anything-plus-model`）、SAM3（`facebook/sam3`，**gated**） |
| 数据 | HF `xg-123/Functional-SLAM-dataset`：18 FunGraph3D + 18 SceneFun3D `rgb_SLAM` 子目录 |
| API | DeepSeek：`DEEPSEEK_API_KEY` + `DEEPSEEK_MODEL`（论文用 DeepSeek-V4-Flash 部署名） |
| 结论 | **已开源**（完整推理管线、配置、数据与文档）。非商用许可；SAM3 与 LLM API 为外部依赖 |

## 入口速查

| 路径 / 命令 | 作用 |
|-------------|------|
| `main.py` | 主入口：跟踪 + 语义/功能图 + 回环 |
| `config/fungraph_eval_node_place.yaml` | 论文完整系统（功能拓扑回环开启） |
| `config/base.yaml` | MASt3R-SLAM 基线；功能回环关闭 |
| `mast3r_slam/tracker.py` | 几何跟踪 |
| `mast3r_slam/semantic/semantic_pipeline.py` | RAM++ / DeepSeek / SAM3 管线 |
| `mast3r_slam/functional_graph/` | 在线功能图：关联、时序后验、回环签名 |
| `logs/<run>/<seq>.txt` | TUM 格式关键帧轨迹 |
| `logs/<run>/semantic/online_functional_graph.json` | 在线功能图 JSON |
| `logs/<run>/*_functional_graph_overlay.ply` | 功能图叠加点云 |

**最短路径：** clone `--recursive` → conda + PyTorch CUDA → `pip install -e` 各子模块 → 下载 checkpoints → `hf download` 数据集 → 设 DeepSeek 环境变量 → `python main.py`（FunGraph3D 或 SceneFun3D 示例见 README）。

**实时配置：** `--semantic-keyframes-only-after-scene-lock`、`--functional-graph-keyframes-only-after-scene-lock`、`--sam3-processor-resolution 672`。

## 对 wiki 的映射

- 论文：[`sources/papers/functional_slam_arxiv_2609_07497.md`](../papers/functional_slam_arxiv_2609_07497.md)
- 沉淀 **[`wiki/entities/paper-functional-slam.md`](../../wiki/entities/paper-functional-slam.md)**

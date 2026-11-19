# AIRSoul / MazeWorld（airs-cuhk/airsoul）

> 来源归档（ingest · 官方源码仓库）

- **名称：** airsoul（`projects/MazeWorld`）
- **类型：** repo
- **URL：** <https://github.com/airs-cuhk/airsoul/tree/main/projects/MazeWorld>
- **配套论文：** [Context and Diversity Matter: The Emergence of In-Context Learning in World Models（arXiv:2509.22353 / ICLR 2026）](../papers/sun_awesome_wm_2509_22353_icl-emergence-context-and-diversity-matt.md)
- **入库日期：** 2026-11-19
- **核对范围：** 通读 `README.md`、`projects/MazeWorld/{README.md, config.yaml, train.py, validate.py, evaluate.py, generate.py, maze_epoch.py}` 与 `airsoul/models/{e2e_objnav_sa.py, decision_model.py}`、`airsoul/modules/blockrec_wrapper.py`、`airsoul/utils/trainer.py`
- **通读：** 已完成
- **一句话说明：** AIRS-CUHK 开源的多范式长上下文具身智能体代码库，其中 `projects/MazeWorld` 与 `E2EObjNavSA` 为 arXiv:2509.22353（L2World）对应世界模型与因果决策官方参考实现。

## 开源状态与目录结构（步骤 2.5）

| 模块 / 入口 | 路径 | 职责与实核要点 |
|-------------|------|----------------|
| 训练入口 | `projects/MazeWorld/train.py` | 实例化 `Runner()`，向 `Runner.start` 注册模型 `E2EObjNavSA` 及两阶段 Epoch 控制器 `[MazeEpochVAE, MazeEpochCausal]` |
| 验证与生成 | `projects/MazeWorld/validate.py`、`generate.py` | `validate.py` 复用 `Runner`；`generate.py` 通过 `GeneratorRunner` 调用 `MAZEGenerator` 离线 Rollout 生成 |
| 交互式评测 | `projects/MazeWorld/evaluate.py` | 加载 `E2EObjNavSA` 权重并在 `l3c.mazeworld`（`mazeworld-discrete-3D-v1`）环境下按 `model.inference_step_by_step` 逐步闭环推理与多步自回归评估 |
| Epoch 与损失 | `projects/MazeWorld/maze_epoch.py` | `MazeEpochVAE`（epoch 0 视觉重建预训练）与 `MazeEpochCausal`（分块切分、`state_dropout=0.20`、`wm-latent` / `wm-raw` / `pm` / `causal-l2` 联合损失） |
| 配置文件 | `projects/MazeWorld/config.yaml` | 显式配置 `rsa_type: podaw`、`memory_type: MEM`、`memory_length: 64` 及序列分段步长 |
| 模型主干 | `airsoul/models/e2e_objnav_sa.py`、`decision_model.py`、`airsoul/modules/blockrec_wrapper.py` | `E2EObjNavSA` 组合 `ImageEncoder`/`ImageDecoder`、`VAE`、`POTARDecisionModel` 与 `BlockRecurrentWrapper` 封装的 `CausalBlock`（GSA）递归记忆状态 |

## 实核差异与使用边界

- `projects/MazeWorld/README.md` 仍标注 `This Directory is Under Construction`，同时核心 `train.py` / `validate.py` / `evaluate.py` / `generate.py` 及 `E2EObjNavSA` 代码已完整可用。
- 运行 `evaluate.py` 与数据集生成依赖外部环境包 `l3c.mazeworld` / `Xenoverse`，不包含独立已打包镜像或直接下载的 320M 帧迷宫原始数据集权重。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2509-22353-icl-emergence-context-and-diversity-matter-the-e.md`](../../wiki/entities/paper-sa-2509-22353-icl-emergence-context-and-diversity-matter-the-e.md)

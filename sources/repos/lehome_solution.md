# lehome_solution（GitHub）

> 来源归档（ingest）

- **项目名称：** lehome_solution — Learning to Fold（LeHome Challenge 2026）
- **GitHub 地址：** <https://github.com/IliaLarchenko/lehome_solution>
- **License：** Apache-2.0
- **Stars（入库快照）：** ~56（2026-07-22）
- **核心功能：** 复现 LeHome 2026 方案的**完整工程**：异步分布式 RL 训练与 rollout、模型训练、自主采集、DAgger/遥操作、Isaac Sim 评测与真机推理。
- **入库日期：** 2026-07-22
- **一句话说明：** 采集 → 训练 → 推理全链路开源仓；仿真权重 [`lehome_sim`](https://huggingface.co/IliaLarchenko/lehome_sim)、真机权重 [`lehome_real`](https://huggingface.co/IliaLarchenko/lehome_real)；博客与 tech report 见 [ilialarchenko.com/projects/lehome2026](https://ilialarchenko.com/projects/lehome2026)。

## 开源边界

| 项 | 状态 |
|----|------|
| 训练 / RL 管线 | **已开源** — `scripts/run_rl_pipeline.py` + `configs/rl_pipeline_sim.yaml` |
| 仿真评测 / 采集 | **已开源** — `scripts/run_eval.py`、`--rollout_worker` |
| 真机 DAgger / 推理 | **已开源** — `scripts/record_real_dagger.py`、`scripts/serve.py` |
| 权重 | **已开源** — HF `lehome_sim` / `lehome_real` |
| 竞赛环境 fork | **已开源** — submodule / 依赖 [`IliaLarchenko/lehome-challenge`](https://github.com/IliaLarchenko/lehome-challenge)（特权仿真数据）；提交用官方仓 |
| 生产就绪度 | README 明确：**竞赛压力代码，宜作参考**；发布前做过重构，遇问题请开 Issue |

## 仓库结构（摘要）

| 路径 | 作用 |
|------|------|
| `scripts/run_rl_pipeline.py` | 分布式 BC→RL 飞轮（`--trainer` / `--rollout_worker`） |
| `scripts/run_eval.py` | Isaac Sim 评测与数据采集（可 `--metrics_only`） |
| `scripts/record_real_dagger.py` | 真机自主 rollout + 人类接管 DAgger |
| `scripts/serve.py` | 真机策略 WebSocket / 服务端 |
| `scripts/dagger_collect.py` | 仿真失败态恢复的 DAgger |
| `scripts/real_camera_align.py` | 相机 overlay 对齐 |
| `scripts/train.py` / `compute_norm_stats.py` / `train_fast_tokenizer.py` | 真机 BC 训练链 |
| `scripts/download_hf_assets.py` / `enrich_garment_type.py` | 拉官方 BC 数据并补衣物类型标签 |
| `configs/` | `rl_pipeline_sim.yaml`、`train_real_bc.yaml`、`real_robot.yaml` 等 |
| `openpi` / `lehome-challenge` | submodules（π₀.₅ 栈 + 环境） |
| `setup.sh` | 系统依赖、venv、Isaac Sim、资产下载 |

## 主入口（README 三工作流）

1. **RL pipeline（仿真）** — HF Hub 作 checkpoint / dataset 总线；Trainer 与多 worker 无屏障异步。
2. **Eval / rollout** — 下载 `lehome_sim` 后 `run_eval.py --all`；期望 seen 衣物约 **80%+** 成功率量级。
3. **Real DAgger** — `serve.py`（主 venv）+ `record_real_dagger.py`（lehome-challenge venv）；`SPACE` 接管、`→` 保存、`←` 丢弃。

## 硬件 / 算力提示

- 训练参考：**1× H200**；大量 rollout 在 **RTX PRO 6000**；磁盘 **500+ GB**（数据+ckpt）。
- 真机：双 leader + 双 follower（间距约 46 cm）、顶视 + 双腕相机；端口写在 `configs/real_robot.yaml`。

## 关联 Wiki / Sources

- [Learning to Fold（论文实体）](../../wiki/entities/paper-lehome-learning-to-fold.md)
- [论文摘录](../papers/lehome_learning_to_fold_arxiv_2606_27163.md)
- [项目博客](../sites/ilialarchenko-lehome2026.md)
- [lehome_sim](../sites/huggingface-lehome-sim.md) · [lehome_real](../sites/huggingface-lehome-real.md)
- [VLA](../../wiki/methods/vla.md) · [LeRobot](../../wiki/entities/lerobot.md) · [DAgger](../../wiki/methods/dagger.md)

## 当前提炼状态

- [x] 仓库入口与三工作流
- [x] 开源/依赖边界
- [x] 与 HF 权重、博客、arXiv 互指

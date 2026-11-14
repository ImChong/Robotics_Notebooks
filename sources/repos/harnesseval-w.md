# HarnessEval-W（MirroS-Lab/HarnessEval-W）

> 来源归档

- **标题：** HarnessEval-W
- **类型：** repo
- **来源：** 镜界（MirroS）/ MirroS-Lab
- **链接：** <https://github.com/mirros-lab/harnesseval-w>（canonical：`MirroS-Lab/HarnessEval-W`）
- **项目页：** <https://mirros-lab.github.io/HarnessEval-W> — 见 [`sources/sites/harnesseval-w-github-io.md`](../sites/harnesseval-w-github-io.md)
- **论文：** <https://arxiv.org/abs/2608.16859>
- **Blog：** <https://mirros.ai/blog/harnesseval>
- **许可：** README 宣称 **Apache-2.0**；GitHub License 元数据截至 2026-08-18 为 `null`
- **语言：** Python；CLI 入口 `harnesseval = harnesseval.cli:main`（`pyproject.toml` 0.1.0）
- **入库日期：** 2026-08-18
- **一句话说明：** 交互式世界模型的 agentic 评测 harness：案例路由 → 技能子代理 → 校验聚合；捆绑 demo 与固定 plans，全量 HF 案例待发。
- **沉淀到 wiki：** [`wiki/entities/paper-harnesseval-w.md`](../../wiki/entities/paper-harnesseval-w.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 待发布 |
|--------|--------|
| 评测代码、`src/harnesseval/skills/`（11 模块）、`pipeline/`（planner/runner）、metric backends | Hugging Face 全量 / 子集案例 |
| `benchmark/plans`、`benchmark/initial_observations`、捆绑 demo `runs/example/results_example` | 托管提交与官方代评服务 |
| CLI：`plan` / `generate` / `eval` / `verify` | GitHub 识别的 LICENSE 元数据（README 已写 Apache-2.0） |

News（README，2026-08-18）：论文、主页、完整评测代码、固定 plans 与 metric backends 已发布。

---

## 仓库入口（README / 目录）

| 组件 | 说明 |
|------|------|
| 安装 | 三个 conda 环境：`harnesseval-main` / `harnesseval-metrics` / `harnesseval-pavrm`（`docs/installation/*.environment.yml`） |
| 凭证 | `cp config/example.env harnesseval.env` |
| 最短评测 | `harnesseval eval --results … --model-id … --run-root … --manifest … --plan-root benchmark/plans` |
| 校验 | `harnesseval verify run --eval-root … --manifest … --model …` |
| 打包 demo | `runs/example/results_example`；`cd` 后可直接 eval + verify |
| 产物 | `evaluation/summary.json`、`leaderboard_latest.json/.csv`、`LEADERBOARD.md` |
| 技能合同 | `src/harnesseval/protocols.py` 的 `SKILLS`（11 个 id） |
| 高层编排 | `src/harnesseval/pipeline/planner.py`、`runner.py`；顶层 `cli.py` 的 `eval` 默认可调 `tools/run_model_eval_pool.sh` |
| 默认清单 | `runs/example/results_example/manifest.json`（当前cli默认；330全量须显式给清单） |
| 致谢 | VBench、WBench、WorldScore、Cosmos、Lingbot World、MiniMax H3 |

### 11 个技能 id（`protocols.py`）

观测侧：`render_quality_inspector`、`motion_quality_inspector`、`appearance_consistency_inspector`、`physical_plausibility_inspector`。

核心 / 诊断：`viewpoint_trajectory_verifier`、`intentional_change_verifier_vlm`、`physical_response_verifier_vlm`、`physical_law_validator`、`drift_degradation_analyzer`、`return_consistency_verifier`、`offscreen_evolution_verifier`。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-harnesseval-w](../../wiki/entities/paper-harnesseval-w.md) | 论文实体、三轴八设定与人对齐数字 |
| [paper-worldscore](../../wiki/entities/paper-worldscore.md) | 开放域相机可控世界生成榜；固定指标 vs agentic harness |
| [ewmbench](../../wiki/entities/ewmbench.md) | 操纵域场景守恒 / 末端 / 语义；轴线不同 |
| [paper-abot-world-0](../../wiki/entities/paper-abot-world-0.md) | Native action 族被评模型之一（Overall 66.1） |

## 固定版本源码核读（2026-11-13）

- **版本：** [ed4ccc6486b8271723ee8baea60d89b32d0a7518](https://github.com/mirros-lab/harnesseval-w/tree/ed4ccc6486b8271723ee8baea60d89b32d0a7518)
- **核对范围：** 根README全文；cli的eval/run-skills/score/verify完整路径；run_model_eval_pool.sh全文及pool.py main/prepare/finalize；runner.plan_skill_tasks/execute_skill_tasks及发布调用链；Intentional.evaluate完整函数；score_components/build_score/execute_score_tasks与aggregate.py全文。
- **通读：** 已完成
- CLI默认eval委托shell→GPU pool，先inventory审核、固定plans展开任务，按skill运行并记录cache/provenance，最后cache-only score/report；verify是单独命令。
- 每case Analyze缓存由初始观测/动作/backend决定，不含模型生成视频；Verify含视频与sampling指纹。同case的分析可跨模型复用；8个子问题由一个Verify响应给出，非8次独立agent调用。
- 当前final_score是0.5×core均值+0.5×observation均值，diagnostic另列；它不是aggregate.py可选evidence cap路径。缺core/observation则不能出完整分，观察质量不只是展示栏。
- 公共100例与贡献页已发布；托管代评待发。README的demo相对plan-root需按仓根运行示例核对，避免从子目录直接照抄造成路径错误。静态核读，未调用付费VLM或跑GPU。

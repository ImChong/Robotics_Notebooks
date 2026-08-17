# PRM-as-a-Judge（过程级机器人评测工具箱）

> 来源归档

- **标题：** PRM-as-a-Judge
- **类型：** repo
- **来源：** PRM-as-a-Judge Team（BAAI / CASIA）
- **链接：** <https://github.com/Yuheng2000/PRM-as-a-Judge>
- **论文：** <https://arxiv.org/abs/2608.14284>（1.5）；README 主引用 1.0 <https://arxiv.org/abs/2603.21669>
- **项目页：** <https://prm-as-a-judge.github.io/>
- **许可：** Apache-2.0
- **入库日期：** 2026-08-17
- **一句话说明：** 用 PRM 把 JSONL manifest 中的 rollout 视频打成进度曲线，再算 OPD 指标并生成交互报告。
- **沉淀到 wiki：** [`wiki/entities/paper-prm-as-a-judge.md`](../../wiki/entities/paper-prm-as-a-judge.md)

---

## 核心定位

配合 *PRM-as-a-Judge 1.5*：不改被评策略，只消费其 rollout 视频，输出过程曲线、指标表（`metrics.xlsx`）与可视化报告。

---

## 仓库入口（README / `eval/`，2026-08-17 核查）

| 组件 | 说明 |
|------|------|
| 安装 | `conda create -n prm-judge python=3.10`；`pip install -e ".[dopamine]" -c constraints/dopamine-cu128-py310.txt` |
| 默认权重 | `hf download tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview` |
| 评测入口 | `MANIFEST=eval/examples/manifest_demo_cases.jsonl PRM_PATH=... VISUALIZE=1 bash eval/run_eval.sh` |
| 报告 | `python eval/run_judge.py serve --run-root eval/results/run_YYMMDD_HHMMSS` |
| Notebook | `getting_started/PRM_as_a_Judge_quickstart.ipynb` |
| 模式 | 默认 `incremental`；可 `EVAL_MODE=forward` / `backward` |
| 自备数据 | JSONL：`case_id` / `task` / `video`；可选 `goal_image`（仅 Robo-Dopamine） |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-prm-as-a-judge](../../wiki/entities/paper-prm-as-a-judge.md) | 1.5 指标、RoboDojo 过程榜与 RoboPulse++ |
| [过程奖励建模](../../wiki/concepts/progress-reward-modeling.md) | PRM 接口与评测透镜 |
| [RoboDojo](../../wiki/entities/robodojo.md) | 被评 rollout 来源榜 |
| [TOPReward](../../wiki/entities/paper-topreward.md) | RoboPulse++ 对照 judge |

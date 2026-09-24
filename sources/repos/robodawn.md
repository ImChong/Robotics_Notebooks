# RoboDawn（Hugo-AGI/RoboDawn）

> 代码仓库归档

- **标题：** RoboDawn — VLM intelligence transfer to robotic control
- **类型：** repo
- **组织：** Hugo-AGI（论文作者维护）
- **代码：** <https://github.com/Hugo-AGI/RoboDawn>
- **许可证：** MIT
- **项目页：** <https://robodawn.top>
- **论文：** <https://arxiv.org/abs/2609.22966>
- **入库日期：** 2026-09-24
- **复核日期：** 2026-09-24
- **一句话说明：** RoboTwin 2.0 + RoboDojo 双 benchmark harness；128 ICL demos、`harness/` 评测脚本、`scripts/robodojo/` VLM 实验；子模块 pinned RoboTwin/RoboDojo。

## 开源状态（README / 仓库结构核查 2026-09-24）

- **已开源：** `harness/`（RoboTwin eval）、`evaluation/policies/vlm_agent/`（RoboDojo policy）、`demos/`（演示库 + MANIFEST）、`scripts/robodojo/`、`tests/`；`secrets.example.json` 模板。
- **复现入口：**
  - RoboTwin：`harness/run_robotwin_eval.py` + `harness/scripts/aggregate_results.py`
  - RoboDojo：`scripts/robodojo/run_vlm_experiment.py` / `run_vlm_eval.sh`
- **依赖：** 外部 VLM API（OpenAI-compatible）；Conda 环境 RoboTwin；子模块 RoboTwin-Platform/RoboTwin、RoboDojo-Benchmark/RoboDojo。
- **资产：** 与论文数字对齐的 seeds、demos；**710** episode 在线回放见 [robodawn.top/results](https://robodawn.top/results)。

## 快速入口

```bash
git clone --recurse-submodules https://github.com/Hugo-AGI/RoboDawn.git
cd RoboDawn
cp secrets.example.json secrets.json  # 填 api_base / api_key
# RoboTwin 单任务评测见 README → Reproducing the RoboTwin 2.0 results
```

## 对 wiki 的映射

- [`wiki/entities/paper-robodawn.md`](../../wiki/entities/paper-robodawn.md)

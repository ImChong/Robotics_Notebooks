# ABot-Navigation（amap-cvlab/ABot-Navigation）

> 来源归档

- **标题：** ABot-Navigation / ABotN-Bench
- **类型：** repo / benchmark-toolkit
- **来源：** 阿里巴巴高德 AMAP CV Lab
- **链接：** <https://github.com/amap-cvlab/ABot-Navigation>
- **默认分支（2026-08-31）：** `ABotN-Bench`（基准与评测）；`main` 仍保留 ABot-N0 介绍
- **项目页：** <https://amap-cvlab.github.io/ABot-Navigation/ABot-N1/> — 见 [`sources/sites/abot-n1.md`](../sites/abot-n1.md)
- **论文：** <https://arxiv.org/abs/2607.10383>
- **许可：** Apache-2.0（`render_server/` 含 3DGS 上游许可）
- **入库日期：** 2026-08-31
- **一句话说明：** ABot-N1 官方开源仓：**ABotN-PointBench / POIBench / Short-Horizon OVON** 数据下载、`abotn_evaluator` 闭环评测、3DGS 多视角 `render_server`；**不含** ABot-N1 策略训练与权重。
- **沉淀到 wiki：** [`wiki/entities/paper-abot-n1.md`](../../wiki/entities/paper-abot-n1.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 未发布（截至 2026-08-31） |
|--------|---------------------------|
| `abotn_evaluator/`（Point / POI / OVON runner） | ABot-N1 慢–快模型 checkpoint |
| `render_server/`（3DGS HTTP 渲染，CUDA 11 独立环境） | 30M 预训练轨迹与 GRPO 训练脚本 |
| `agent_examples/`、`docs/`、`scripts/` 启动脚本 | 论文报告的统一推理 demo |
| HF / ModelScope 三套 benchmark 数据 | — |

---

## 仓库入口（`ABotN-Bench` 分支）

| 组件 | 说明 |
|------|------|
| 安装 | `git clone … && pip install -e .` → `import abotn_evaluator` |
| 渲染服务 | `conda env create -f render_server/environment.yml`；设 `SCENES_ROOT` 后 `bash scripts/start_PointGoal_outdoor_render_server.sh` |
| Point-Goal 评测 | `python -m abotn_evaluator.point_goal.runner --agent-module your_agent:YourAgent --data-dir …/annotations --render-url http://localhost:7036/render_gs --mode outdoor` |
| POI-Goal 评测 | 见 `docs/poi-goal.md` |
| Short-Horizon OVON | Habitat-sim 集成，见 `docs/short-horizon-ovon.md` |
| Agent 接口 | 实现 `BasePointGoalAgent.reset()` + `predict(Observation) -> WaypointPrediction` |

数据集布局（README）：`ABotN-PointBench/{Indoor,Outdoor}/{annotations,occmaps}/`；`ABotN-POIBench/{annotations,occmaps}/`；3DGS `.ply` 单独目录作 `SCENES_ROOT`。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-abot-n1](../../wiki/entities/paper-abot-n1.md) | 论文实体、慢–快架构与结论 |
| [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md) | VLN 任务总览 |
| [paper-abot-m05](../../wiki/entities/paper-abot-m05-mobile-manipulation-wam.md) | 同机构 ABot 家族，移动操作 WAM |
| [paper-abot-world-0](../../wiki/entities/paper-abot-world-0.md) | 同机构交互式世界模型 |

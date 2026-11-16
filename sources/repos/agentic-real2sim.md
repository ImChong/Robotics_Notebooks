# Agentic Real2Sim 官方源码

> 来源归档（官方 GitHub）

- **标题：** Agentic Real2Sim: Physics-based World Modeling with Vision-Language Agents
- **类型：** repo / Real2Sim / VLM agents / MuJoCo
- **代码：** <https://github.com/agentic-real2sim/agentic_real2sim>
- **项目页：** <https://agentic-real2sim.github.io/>
- **论文：** <https://arxiv.org/abs/2607.19190>（v5，2026-10-11）
- **核查版本：** [`880389b`](https://github.com/agentic-real2sim/agentic_real2sim/tree/880389bda289d2e78154c73ae1b6d23b7a650557)
- **核对范围：** 官方 README、`ar2s/run_pipeline.py` 入口、安装/运行/产物说明（静态阅读；未执行整个管线）。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **一句话说明：** 由固定顺序的 Python runner 调度视觉、物性、场景和抓取优化工具，把机器人录像转换为 MuJoCo episode twin。

## 安装与运行入口

官方 README 要求 Linux、NVIDIA GPU、Python 3.12+、Conda 和 C/C++ 编译工具；约 80 GB 磁盘工作空间。模型权重未全部随仓发布，部分 SAM3 / SAM3D checkpoint 需要接受授权并设置 `HF_TOKEN`。

```bash
# 先按官方 README 安装环境并配置模型凭据
python -m ar2s.run_pipeline \
  --raw-data data/airbot/raw_data/AIRBOT_success_2026_08_03_21_21_47 \
  --agent-config ar2s/agent_configs/config_orouter_qwen.yaml \
  --artifact-root outputs/run_pipeline
```

自采 AIRBOT Play 数据可用 `scripts/export_collected_episode.py` 将 ZED SVO2 + MCAP + 标定元数据整理成 DROID-shaped `raw_data`。支持 `--start-from` 与 `--until` 分阶段复跑。

## 源码结构与运行顺序

`ar2s/run_pipeline.py` 中 `_STAGE_ORDER` 固定：

```text
visual_processing -> geometry_prior -> scene_view_repair
-> physical_prior -> scene_prep -> grasp_optimization
```

| 模块 | 作用 |
|---|---|
| `ar2s/run_pipeline.py` | 依次调度六阶段，VLM 决策在子代理中 |
| `ar2s/agents_visual/` | 视频、分割、mesh、pose 与 episode bundle |
| `ar2s/agents_geometry_prior/` | 方向、尺度、地面先验 |
| `ar2s/agents_visual/subagents/scene_view_repair` | 次视角修复缺失物体或场景不一致 |
| `ar2s/agents_physical_prior/` | 材料与质量先验 |
| `ar2s/agents_sysid/` | 场景标定、抓取 sweep/loop |
| `ar2s/droid_sim/` | MuJoCo 场景及 USD 输出 |

产物包括 `outputs/run_pipeline/<episode_id>/sysid_inputs/manifest.yaml`、`objects/`、`physical_priors.json`、`calibration.yaml` 与 `artifacts/sweep-*/`。

## 边界

官方开源的是转换管线，不等于所有模型权重均可直接下载；也不能据此断言可变形与人形实验全部可复现。此次仅静态核查，未进行 GPU 环境部署和性能指标复现。

## 对 wiki 的映射

- [Agentic Real2Sim 论文实体](../../wiki/entities/paper-agentic-real2sim.md)
- [论文归档](../papers/agentic_real2sim_arxiv_2607_19190.md)
- [项目页归档](../sites/agentic-real2sim-github-io.md)

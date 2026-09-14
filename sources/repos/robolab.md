# RoboLab（NVLabs 官方仓库）

> 来源归档

- **标题：** RoboLab
- **类型：** repo / benchmark toolkit
- **组织：** NVLabs（NVIDIA）
- **代码：** <https://github.com/NVLabs/RoboLab>
- **主页：** <https://research.nvidia.com/labs/srl/projects/robolab/>
- **论文：** <https://arxiv.org/abs/2604.09860>
- **Stars：** ~496（2026-09-14）
- **License：** Apache-2.0
- **技术栈：** Python 3.11 · uv · Isaac Sim 5.0/5.1 · Isaac Lab 2.2/2.3
- **入库日期：** 2026-09-14
- **一句话说明：** 基于 Isaac Lab 的通用操纵策略仿真评测：RoboLab-120 任务、server-client 策略架构、多环境并行、Claude Code skills 生成场景/任务、自包含结果 Dashboard。

## 仓库结构（README）

```text
assets/objects/          物体库（312+）
assets/scenes/             场景 USD
robolab/tasks/             RoboLab-120 任务定义
robolab/robots/            机器人配置
policies/                  策略客户端（pi0_family、volo 等）
skills/robolab-scenegen/   场景生成 skill
skills/robolab-taskgen/    任务生成 skill
examples/                  空跑、回放、夹爪测试
docs/                      Dashboard、调试、生态文档
```

## 运行要点（策展）

| 项 | 说明 |
|----|------|
| 安装 | `uv sync --extra isaac50` 或 `isaac51`；两栈不可共存于同一 venv |
| 快速验证 | `uv run pytest tests/`（含单 episode 端到端） |
| 策略评测 | `uv run python policies/pi0_family/run.py --policy pi05 --task BananaInBowlTask --num-envs 10` |
| 架构 | 策略作独立 server；RoboLab 通过轻量 client 连接 |
| 并行 | `--num-envs N` 向量化多 episode |
| 生态 | [RoboVoLo](https://github.com/NVlabs/RoboVoLo) 扩展长程推理任务 |

## 对 wiki 的映射

- 实体页：[RoboLab](../../wiki/entities/robolab.md)
- 项目页：[robolab-nvidia.md](../sites/robolab-nvidia.md)
- 论文：[robolab_arxiv_2604_09860.md](../papers/robolab_arxiv_2604_09860.md)

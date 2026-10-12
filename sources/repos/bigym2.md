# BiGym2（swirl-uk/BiGym2）

> 来源归档

- **标题：** BiGym 2.0
- **类型：** repo
- **来源：** Imperial College London（SWIRL）
- **链接：** <https://github.com/swirl-uk/BiGym2>
- **项目页：** <https://bigym2.github.io/>
- **论文：** <https://arxiv.org/abs/2610.07594>
- **入库日期：** 2026-10-12
- **许可：** Apache-2.0（GR00T-WBC 权重为 NVIDIA Open Model License，见仓库 NOTICE）
- **一句话说明：** MuJoCo 中 Unitree G1 行走家庭操作基准：20 任务、冻结 GR00T-WBC 下身、示范下载/查看/重渲染、VR 采集、编码智能体评测 CLI。
- **沉淀到 wiki：** [`wiki/entities/paper-bigym2-humanoid-household-benchmark.md`](../../wiki/entities/paper-bigym2-humanoid-household-benchmark.md)

## 目录与入口（2026-10-12 浅克隆核对）

| 路径 | 说明 |
|------|------|
| `bigym/envs/` | 任务环境（reach_target.py、move_plates.py、manipulation.py、dishwasher*.py、cupboards.py 等）与 G1 XML |
| `bigym/loco/` | 行走版核心：env.py、controller.py、lowerbody.py、command.py、tasks.py、eval/、demos/、agent/、adapters/（GR00T-WBC） |
| `bigym/vr/` | Quest 3 VR：collect/（`bigym-collect`）、ik/、viewer/（`bigym-view`） |
| `bigym/robots/` | 机器人配置与夹爪 |
| `examples/` | `train_act.py`（简化 ACT，README 说明不复现论文 ACT 结果）、`loco_random_agent.py` |
| `docs/`、`tests/` | 文档与任务测试 |

`pyproject.toml` 声明命令：`bigym-collect`、`bigym-view`、`bigym-download`、`bigym-export-lerobot`、`bigym-rerender-lerobot`、`bigym-agent`。安装用 `uv sync`（Python 3.10+）。

## 关键用法（README）

```bash
uv run bigym-download --task move_plate pick_box      # 下载示范
uv run bigym-view --task move_plate                   # 浏览器 3D 查看
uv run --with torch --with imageio-ffmpeg python examples/train_act.py --task move_plate --record 3
uv run bigym-agent run --task move_plate --harness codex --model gpt-6-astra --effort high
uv run bigym-collect --task move_plate                # Quest 3 采集（Linux）
```

示范全量约 29 GB，仅下载智能体所需文件（`--agent`）约 68 MB。News：2026/10/08 发布代码、1,200 条示范和 5,400 条智能体回合。

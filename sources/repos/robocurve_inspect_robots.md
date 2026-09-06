# robocurve/inspect-robots

> 来源归档

- **标题：** Inspect Robots — open-source evaluation framework for physical AI
- **类型：** repo
- **组织：** Robocurve
- **代码：** <https://github.com/robocurve/inspect-robots>
- **文档：** <https://docs.inspectrobots.org/>
- **LLM 文档：** <https://docs.inspectrobots.org/llms.txt>
- **Stars：** ~298（2026-09-06）
- **License：** MIT
- **Python：** 3.10–3.13
- **状态：** alpha（API 可能变；README 建议 pin 版本）
- **入库日期：** 2026-09-06
- **一句话说明：** **Physical AI 真机优先评测框架**：一次定义 benchmark，任意 **Policy（VLA/LLM/CaP）× Embodiment（真机/仿真）** 组合，产出可审计 **EvalLog** + **Rerun** 可视化；对标 [Inspect AI](https://inspect.aisi.org.uk/) 的机器人版。
- **沉淀到 wiki：** [`wiki/entities/inspect-robots.md`](../../wiki/entities/inspect-robots.md)、[`wiki/entities/robocurve.md`](../../wiki/entities/robocurve.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **框架本体** | **已开源**（MIT） |
| **插件** | 本仓 first-party 插件 + 独立 rig 包（yam/franka/g1/agibot/so101/widowx/ros/isaacsim/xpolicylab/agent/capx/voice） |
| **Benchmark 任务** | 具体 benchmark 在 [WorldEvals](https://github.com/robocurve/worldevals) 等独立仓 |
| **被测模型权重** | 不随框架分发；经 OpenPI、XPolicyLab server、Anthropic API 等接入 |
| **Rerun** | 可选 extra `inspect-robots[rerun]` |

## README 要点（2026-09-06）

### 安装

```bash
uv venv && uv pip install "inspect-robots[rerun]"
# 或 numpy-only 核心：uv pip install inspect-robots
```

### Quickstart（摘录）

```bash
uv pip install inspect-robots-yam
inspect-robots setup          # 写 ~/.config/inspect-robots/config.ini
inspect-robots "place the fork on the plate"
```

### Policy 族

| `--policy` | 说明 |
|------------|------|
| `agent` | 前沿 LLM 经 tool call 驱动（Claude/GPT 等） |
| `xpolicylab` | XPolicyLab 上 40+ VLA（π0、GR00T、OpenVLA-OFT、SmolVLA…） |
| `capx` | CaP-X 代码即策略（SAM3 + Contact-GraspNet + Pyroki） |
| `scripted` / rig 自带 | MolmoAct2、OpenPI、LeRobot 等（见各 rig README） |

### Embodiment 族（摘录）

| 真机 | `--embodiment` | 包 |
|------|----------------|-----|
| I2RT YAM 双臂 | `yam_arms` | inspect-robots-yam |
| Franka FR3/Panda | `franka` | inspect-robots-franka |
| AgiBot A2 Ultra | `a2_arms` | inspect-robots-agibot-a2 |
| Unitree G1 臂 | `g1_arms` | inspect-robots-unitree-g1 |
| SO-ARM100/101 | `so_arm` | inspect-robots-so101 |
| WidowX 250S | `widowx` | inspect-robots-widowx |
| 任意 ROS 臂 | `ros` | inspect-robots-ros |
| Isaac Lab | `isaacsim` | inspect-robots-isaacsim |
| Mock | `cubepick` | 内置 |

### 设计原则

- **Real-world first** — 人机 reset、无特权 success oracle、墙钟控制率
- **Compatibility checked up front** — `(policy, embodiment)` 动作/观测语义 upfront 校验
- **Auditable EvalLog** — schema 版本化、git revision、包版本；可离线 re-score
- **Light core** — 仅依赖 NumPy；Rerun/仿真/VLA 为可选插件
- **Safe unattended** — 显式 error taxonomy；故障不自动过夜推进
- **VLA-native** — action chunking、ACT/ALOHA temporal ensembling、action semantics

### Inspect AI 映射

| Inspect AI | Inspect Robots |
|------------|----------------|
| `Model` | `Policy` + `Embodiment` |
| `Task = dataset + solver + scorer` | `Task = scenes + controller + scorer` |
| `Sample` | `Scene` |
| `eval()` → `EvalLog` | `eval()` → `EvalLog` |

### CLI 工具链

- `inspect-robots list` — 注册 task/policy/embodiment
- `inspect-robots inspect LOG.json` — 打印日志
- `inspect-robots summarize` — 失败 run → learnings markdown
- `inspect-robots view logs/` — HTML 报告与索引
- `inspect-robots video` — `--store-frames` run 导出 MP4

## 对 wiki 的映射

- [Inspect Robots](../../wiki/entities/inspect-robots.md)
- [Robocurve](../../wiki/entities/robocurve.md)
- [XPolicyLab](../../wiki/entities/xpolicylab.md)
- [Isaac Lab-Arena](../../wiki/entities/isaac-lab-arena.md)
- [LeRobot](../../wiki/entities/lerobot.md) — SO-ARM `lerobot` policy 插件
- [具身评测基准选型闭环](../../wiki/overview/hub-embodied-eval-benchmark.md)

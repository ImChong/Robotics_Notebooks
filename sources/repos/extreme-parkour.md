# extreme-parkour

> 来源归档

- **标题：** extreme-parkour
- **类型：** repo
- **来源：** chengxuxin（CMU / Pathak Lab）
- **链接：** <https://github.com/chengxuxin/extreme-parkour>
- **Stars：** ~1.1k（2026-06）
- **入库日期：** 2026-06-04
- **一句话说明：** ICRA 2024 **Extreme Parkour** 官方实现：基于 **Isaac Gym Preview 3/4 + legged_gym + rsl_rl** 的两阶段跑酷训练栈（特权 base → 深度 + 航向蒸馏），含 JIT 导出与 web viewer。
- **沉淀到 wiki：** [`wiki/entities/extreme-parkour.md`](../../wiki/entities/extreme-parkour.md)

---

## 核心定位

开源 **四足感知跑酷** 复现入口：在 ETH **legged_gym** 范式上扩展 **parkour 地形、统一奖励、双重蒸馏与深度输入**，目标平台为 **Unitree Go1** 级四足（论文强调低成本硬件 + 单目深度）。

## 依赖栈

| 组件 | 版本 / 说明 |
|------|-------------|
| Python | 3.8（conda `parkour`） |
| PyTorch | 1.10.0 + cu113 |
| Isaac Gym | Preview 3 训练；Preview 4 可用 |
| legged_gym | 仓库内 `legged_gym/` 子模块式安装 |
| rsl_rl | 仓库内 `rsl_rl/` |
| 其他 | `numpy<1.24`, pydelatin, wandb, opencv-python, pyfqmr, flask |

## 仓库结构（要点）

```
extreme-parkour/
├── legged_gym/          # 环境、奖励、parkour 地形与训练脚本
│   └── scripts/
│       ├── train.py     # Phase 1 base / Phase 2 distillation（--use_camera）
│       ├── play.py      # 仿真回放
│       └── save_jit.py  # 导出 traced 模型供部署
├── rsl_rl/              # PPO 训练器（fork）
└── images/              # teaser 等
```

## 训练流程（README）

1. **Base policy（Phase 1）：** `python train.py --exptid xxx-xx-WHATEVER --device cuda:0`  
   - 10–15k iterations，3090 约 8–10 h（建议 ≥15k）
   - 可选 `--delay`（8k iter 后加延迟，更贴近实机）

2. **Distillation（Phase 2）：** `python train.py --exptid yyy-yy-WHATEVER --device cuda:0 --resume --resumeid xxx-xx --delay --use_camera`  
   - 5–10k iterations，5–10 h（建议 ≥5k）
   - `--use_camera` 启用深度输入；`--resumeid` 指向 Phase 1 run

3. **部署导出：** `save_jit.py --exptid xxx-xx` → `legged_gym/logs/parkour_new/xxx-xx/traced/`

## 关键 CLI 参数

| 参数 | 含义 |
|------|------|
| `--exptid` | 实验 ID（`xxx-xx` 前缀需唯一，parser 自动前缀匹配） |
| `--use_camera` | 深度相机 vs scandots |
| `--delay` | 观测–动作延迟（蒸馏与 8k+ play 常用） |
| `--resume` / `--resumeid` | 从 Phase 1 checkpoint 继续蒸馏 |
| `--web` | headless 机器上 VS Code Live Preview 端口转发可视化 |

## 与本仓库其他资料的关系

| 资料 | 关系 |
|------|------|
| [legged_gym.md](legged_gym.md) | 上游训练框架；本仓库 fork 并扩展 parkour 任务 |
| [extreme_parkour_arxiv_2309_14341.md](../papers/extreme_parkour_arxiv_2309_14341.md) | 配套论文摘录 |
| [extreme-parkour-github-io.md](../sites/extreme-parkour-github-io.md) | 项目页视频与 ablation |
| Robot Parkour Learning（ZiwenZhuang/parkour） | 同期 Teacher–Student 四足跑酷姊妹线（Unitree 官方生态） |

## 为何值得保留

- **可复现的端到端视觉跑酷基线**：社区大量 parkour / 感知 loco 工作引用的 **ICRA 2024** 开源栈。
- **工程细节完整**：两阶段训练命令、JIT 导出、web viewer 降低复现门槛。
- **与 privileged-training 叙事一致**：scandots + oracle heading → 深度 + 自预测航向的标准范例。

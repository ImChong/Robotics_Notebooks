# DemoGrasp 仓库核查

> 来源归档（repo / BeingBeyond GitHub）

- **URL：** <https://github.com/BeingBeyond/DemoGrasp>
- **机构 / 个人：** BeingBeyond（智在无界）
- **仓库核查commit：** `1255dc5`（180 个文件，含 127 个 `.py`/`.sh` 源码与配置）
- **开源结论：** **已开源可执行 IsaacGym 仿真训练、编辑回放与视觉蒸馏相关源码**。
- **核对范围：** 已通过 `gh api` 核对 `BeingBeyond/DemoGrasp @ 1255dc5` 完整文件树与 `README.md` 入口命名（`scripts/`、`algo/`）；未实际运行 IsaacGym GPU 仿真与真机脚本。
- **通读：** 已完成

## 核心源码入口与模块

- **RL 策略训练与数据重放**：`scripts/` 下实验配置与运行入口，配合 `algo/pn_utils/maniskill_learn/apis/train_rl.py` 执行 1-step MDP PPO（7,000 并行环境）编辑手腕 SE(3) 位姿与手指关节增量
- **视觉/状态骨干网络**：`algo/pn_utils/act.py` 与 `algo/pn_utils/maniskill_learn/networks/`（PointNet、Transformer、MLP、VAE 等策略与价值网络组件）
- **与论文对应**：对应 `arXiv:2509.22149`（ICLR 2026）单演示抓取编辑及闭环策略管线。

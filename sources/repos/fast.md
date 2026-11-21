# FAST 仓库核查

> 来源归档（repo / BeingBeyond GitHub）

- **URL：** <https://github.com/BeingBeyond/FAST>
- **机构 / 个人：** BeingBeyond（智在无界）
- **仓库核查commit：** `20c8537`（33 个文件，仅 `README.md` 与 `docs/` 静态项目页资产，`.py`/`.sh` 可运行代码数为 0）
- **开源结论：** **未开源可执行代码与模型权重**（仓库仅托管项目主页与静态图，无 IsaacGym/MuJoCo MoE-PPO 或 Parseval 残差微调代码；已据此修正早期自动归档对“已开源”的误标）。
- **核对范围：** 已通过 `gh api` 拉取并核对 `BeingBeyond/FAST @ 20c8537` 完整文件树与 `README.md`；无源码可供执行流分析。
- **通读：** 已完成

## 核心摘要

- 提供博客 <https://research.beingbeyond.com/fast>、项目主页 <https://beingbeyond.github.io/FAST/> 与论文 `arXiv:2602.11929`（CoRL 2026）索引。
- 截至 2026-11-21 尚未包含训练与 Unitree G1 部署代码。

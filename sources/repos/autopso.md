# EMI-Group/AutoPSO

- **标题：** AutoPSO 官方实现
- **类型：** repo
- **URL：** <https://github.com/EMI-Group/AutoPSO>
- **许可：** 仓内无根级 SPDX LICENSE
- **配套论文：** [arXiv:2608.07539](https://arxiv.org/abs/2608.07539) — [`sources/papers/autopso_arxiv_2608_07539.md`](../papers/autopso_arxiv_2608_07539.md)
- **入库日期：** 2026-08-18

## 一句话说明

EvoX + PyTorch 双层 PSO：外层搜组件，内层广义 PSO 求解 CEC2022 等任务。

## 仓库状态（2026-08-18 核查）

| 项 | 内容 |
|----|------|
| 包 | `src/autopso/` |
| 示例 | `examples/pytorch/example_cec2022.py` |
| 安装 | `pip install -e ".[pytorch]"`；GPU 需先装匹配 CUDA 的 PyTorch + `evox` |
| 环境检查 | `show_env.py` |

最短复现：装 PyTorch/EvoX → `pip install -e ".[pytorch]"` → `python examples/pytorch/example_cec2022.py`（墙钟实验按 README 需 GPU）。

## 与 wiki 的关系

- 实体页：[paper-autopso](../../wiki/entities/paper-autopso.md) — 含源码运行时序图。

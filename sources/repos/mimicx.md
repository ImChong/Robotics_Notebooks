# MimicX（官方源码仓库）

> 来源归档（核查日期：2026-10-12）

- **类型：** repo / humanoid motion tracking / policy refinement
- **链接：** <https://github.com/NEBULIS-Lab/MimicX>
- **项目页：** <https://nebulis-lab.com/MimicX>
- **论文：** [arXiv:2610.09055](https://arxiv.org/abs/2610.09055)
- **策略：** <https://huggingface.co/Shuaijun/MimicX-Policies>
- **数据与资产：** <https://huggingface.co/datasets/Shuaijun/MimicX-Assets>
- **许可：** Apache-2.0（MimicX 代码）；第三方依赖、数据和可视化资产依各自许可
- **一句话说明：** 公开实现视频重建到 G1 动作跟踪，并将 rollout 诊断、监督候选、继续训练与重复执行门控组成可恢复的精炼闭环。

## 仓库组成与运行入口

| 路径 / 模块 | 用途 |
|-------------|------|
| `mimicx/refinement/` | 诊断条件搜索、候选验证与持久化状态 |
| `mimicx/runtime/` | 依赖感知异构执行与结果等价核验 |
| `mimicx/adapters/` | 动作格式适配 |
| `mimicx/evaluation/` | 指标归约与实验结果工具 |
| `mimicx/visualization/` | rollout 回放、场景对象注册及相机 / 几何工具 |
| `scripts/` | 环境准备、格式转换、训练闭环与导出入口 |
| `mimicx/configs/paper/` | 冻结方法设置与校验和输入清单 |
| `tests/` | 可在 CPU 上运行的回归测试 |

README 基础安装路径为 Python 3.11 虚拟环境，安装 `.[dev,visualization]` 并运行 `python -m pytest -q`。CPU fixtures 可验证确定性 refinement cycle，不需要 SMPL 模型或预训练策略；实际 MuJoCo / MjLab 训练评估还需安装固定的 tracking backend，并准备获授权输入资产。入口见官方 [Quickstart](https://github.com/NEBULIS-Lab/MimicX#quickstart) 与 [Documentation](https://github.com/NEBULIS-Lab/MimicX#documentation)。

## 对 wiki 的映射

- [MimicX 论文与项目实体](../../wiki/entities/paper-mimicx.md)
- [项目页来源](../sites/mimicx-nebulis-lab.md)
- [论文来源](../papers/mimicx_arxiv_2610_09055.md)

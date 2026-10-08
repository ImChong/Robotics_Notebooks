# PredActor（官方评测代码仓）

> 来源归档（最近复核：2026-10-08）

- **类型：** repo / evaluation code / MuJoCo
- **链接：** <https://github.com/MasterYip/PredActor>
- **项目页：** <https://masteryip.github.io/predactor.github.io/> — [团队与联系](https://masteryip.github.io/predactor.github.io/#people)
- **论文：** [arXiv:2609.24840](https://arxiv.org/abs/2609.24840)
- **公开评测入口：** `predactor-eval`（`PredActor/cond_eval.py`）
- **许可：** MIT（仓库）；机器人资产与第三方依赖仍受各自上游许可约束
- **Python：** 3.10（README quick-evaluation 环境）
- **入库日期：** 2026-09-24
- **版本快照：** `pyproject.toml` 0.1.0；作为项目包装号记录，不代表论文方法版本
- **一句话说明：** 官方仓已从占位 README 更新为可运行的 bounded MuJoCo evaluation release，可加载公开 PDP051 与 G1 MotionCLIP checkpoint；训练、数据采集与真机部署尚未公开。

## 当前公开范围（2026-10-08）

| 能力 | 状态 |
|------|------|
| 浏览器式 MuJoCo 评测 | **已开放** — evaluator、G1 评测资产和配置 |
| 评测 checkpoint | **已开放** — [PredActor_Artifacts](https://huggingface.co/MasterYip/PredActor_Artifacts)，见[单独归档](predactor-artifacts.md) |
| 训练数据采集/预处理/标注 | **未开放** — README release checklist 尚未勾选 |
| BC 训练与复现材料 | **未开放** |
| DAgger 数据聚合/策略精炼 | **未开放** |
| G1 机载部署/硬件控制工具 | **未开放** |

## 快速评测路径

官方 README 给出的 Linux 流程：

```bash
git clone https://github.com/MasterYip/PredActor.git
cd PredActor
uv sync --locked
uv run --locked python scripts/hf_download.py --filter checkpoints
uv run --locked predactor-eval
```

下载器从 Hugging Face 拉取模型卡标注的 checkpoint，并写入本地 `Artifacts/`；评测器打开 `http://127.0.0.1:8765/`，有 CUDA 时使用 CUDA，否则回退到 CPU。仓库依赖锁定到 Python 3.10；不要求安装 Isaac Sim。

## 复现与部署边界

- 这是一条带公开 checkpoint 的 **MuJoCo evaluation path**，不是完整训练发布，也不包含论文中动作库、数据采集、teacher rollout、DAgger 训练所需数据和脚本。
- 仓库 README 明确将 BC training、DAgger、data collection/labeling 和 hardware deployment 列为后续 release 项。
- 评测能够验证软件、配置与 checkpoint 的可加载性；不构成机器人硬件安全或真机运动质量保证。
- checkpoint 用 PyTorch pickle-compatible 反序列化；应通过官方 downloader 获取并核对 SHA-256。

## 对 wiki 的映射

- [PredActor 论文实体](../../wiki/entities/paper-predactor.md)
- [评测 checkpoint 归档](predactor-artifacts.md)
- [项目页归档](../sites/predactor-masteryip-github-io.md)
- [论文来源归档](../papers/predactor_arxiv_2609_24840.md)
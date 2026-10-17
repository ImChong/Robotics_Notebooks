# Cepheid（Su-Terry/cepheid）

- **类型：** GitHub 仓库
- **仓库：** <https://github.com/Su-Terry/cepheid>
- **论文：** <https://arxiv.org/abs/2610.18871>（归档：[cepheid_arxiv_2610_18871.md](../papers/cepheid_arxiv_2610_18871.md)）；项目页：[cepheid-project.md](../sites/cepheid-project.md)
- **源码状态：** 已公开
- **许可证：** 代码 MIT；Hugging Face 权重仅限非商用研究（README / 模型卡所述）
- **核验日期：** 2026-10-17
- **知识节点：** [Cepheid](../../wiki/entities/paper-cepheid.md)

## 仓库说明

稀疏顶点位移到稠密运动补全：谐波延拓、冻结岭回归幅值规则、流匹配残差网络。README 含 Quick start（`Cepheid.from_pretrained`、`complete()`）、模型库、局限与引用。

## 结构与入口

| 路径 | 作用 |
|------|------|
| `cepheid/`（`api.py`、`examples.py`、`assets/`） | 库本体 |
| `scripts/train.py` | 按 YAML 训练一个模型 |
| `scripts/run_matrix.py` | 论文表 2 / 4 / 8 评分与汇总 |
| `reproduce/` | 其余表格复现（表 1 / 3 / 14 / 15） |
| `tools/verify_release.py` | 对照论文数值校验发布 |
| `configs/` | 每个检查点一份 YAML |

依赖：numpy、scipy、PyYAML、safetensors、huggingface_hub；PyTorch 自装；复现用 `environment.yml`。

## 访问与复现边界

未随附：DOMA 与 NSDP 基线、GenBot1K 回放（需外部代码）；数据集不再分发（见 `DATA.md`）；表 14 重评分所需脚本不在发布内。README 称默认约 100 s（CPU）/ 3 s（RTX 4090）。本次未运行。

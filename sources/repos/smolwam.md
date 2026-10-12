# SmolWAM（iitimii/SmolWAM）

> 来源归档

- **标题：** SmolWAM
- **类型：** repo
- **来源：** ML Collective（Timi Owolabi 等）
- **链接：** <https://github.com/iitimii/SmolWAM>
- **代码：** <https://github.com/iitimii/SmolWAM>
- **论文：** <https://arxiv.org/abs/2610.13740>（见 [smolwam_arxiv_2610_13740.md](../papers/smolwam_arxiv_2610_13740.md)）
- **项目页：** <https://smolwam.vercel.app/>（见 [smolwam-vercel-app.md](../sites/smolwam-vercel-app.md)）
- **权重：** <https://huggingface.co/timiiowolabi/SmolWAM>
- **许可：** Apache-2.0
- **入库 / 核查日期：** 2026-10-12（浅克隆 tip `dde1e5b`，提交日 2026-10-09）
- **沉淀到 wiki：** [`wiki/entities/paper-smolwam.md`](../../wiki/entities/paper-smolwam.md)

## 仓库结构（核查）

| 路径 | 作用 |
|------|------|
| `smolwam/policy.py` | `SmolWAM.from_pretrained`、`encode_instruction`、`encode_observation`、`predict_action(return_future)`、`decode_future` |
| `smolwam/model.py`、`nn/{dit,wan_vae,adapter}.py` | `SmolWAMModel`（剪枝 DiT 主干、`TextUpdate`、`pad_context`、`load_cosmos_trunk`）、Wan VAE、潜变量适配器 |
| `smolwam/layout.py`、`text.py`、`preprocessing.py` | 紧凑 token 布局、缓存 BERT 文本、图像预处理 |
| `smolwam/train/{kd_targets,bank,adapter,data,train}.py` | 教师目标缓存、数据 bank、适配器与蒸馏训练 |
| `smolwam/eval_libero.py`、`envs/libero.py` | LIBERO 评测（`--protocol full|screen|smoke`） |
| `smolwam/bench.py`、`world_eval.py` | 延迟/FLOPs 与世界头评测 |
| `examples/{quickstart,world_model,libero_rollout}.py` | 示例 |
| `experiments/*.sh` | 论文数字复现脚本：`train_smolwam.sh`、`eval_libero.sh`、`bench.sh`、`world_eval.sh` |

## 开放范围

- 代码、权重、论文 PDF（`paper/smolwam.pdf`）、复现脚本齐全；README 给出 Apple M2 Pro（mps）单回合示例与预期数值表（A100-40GB，PyTorch 2.7 / CUDA 12.8，训练约 5 h 含数据流水线）。
- 评测教师需上游 `NVlabs/cosmos-policy`；训练只需下载教师权重与缓存 T5 嵌入。

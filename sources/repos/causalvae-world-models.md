# Dzyy123/CausalVAE-World-Models

> 来源归档

- **标题：** CausalVAE as a Plug-in for World Models 官方实现
- **类型：** repo
- **代码：** <https://github.com/Dzyy123/CausalVAE-World-Models>
- **License：** MIT（fork [CausalMBRL](https://github.com/dido1998/CausalMBRL) / [C-SWM](https://github.com/tkipf/c-swm)）
- **论文：** <https://arxiv.org/abs/2604.07712>
- **入库日期：** 2026-09-09
- **一句话说明：** CausalVAE 外挂因果层 + 三阶段训练 + 反事实评测脚本；Physics/Chemistry 环境继承 CausalMBRL。

## 开源核查（2026-09-09）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** · MIT |
| Checkpoints | **未随仓发布**（README 说明） |
| 环境 | `environment_py37.yml` / `setup_env.sh` |

## 入口速查

| 命令 | 作用 |
|------|------|
| `bash scripts/run_stage1_Modular_Contrastive.sh` | Stage 1 骨干预训 |
| `bash scripts/run_cswm_causalvae_stage3.sh` | Stage 2–3 因果分支 + 融合 |
| `python experiments/run_causal_eval.py` | 反事实检索评测 |
| `python scripts/eval_counterfactual_cswm_causalvae.py` | C-SWM + CausalVAE 反事实 |

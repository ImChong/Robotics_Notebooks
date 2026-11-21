# Rethink_VLA 仓库核查

> 来源归档（repo / BeingBeyond GitHub）

- **URL：** <https://github.com/BeingBeyond/Rethink_VLA>
- **机构 / 个人：** BeingBeyond（智在无界）
- **仓库核查commit：** `d3739ab`（65 个文件，含 56 个 `.py`/`.sh`/`.yaml` 源码与配置）
- **开源结论：** **已开源完整预训练、微调、推理服务、评测脚本及 Hugging Face 权重**（`wwwyyy/pretrain-oxe-eef-relative`、`wwwyyy/pretrain-RealSimEEFJoint-eef-relative`）。
- **核对范围：** 已通过 `gh api` 核对 `BeingBeyond/Rethink_VLA @ d3739ab` 完整文件树、`README.md` 并直接阅读 `shell/pretrain-M1-240k.sh` 源码配置；未实际启动多卡 GPU 训练与真机测试。
- **通读：** 已完成

## 核心源码入口与模块

- **预训练脚本与主入口**：`shell/pretrain-M1-240k.sh` / `pretrain-M1-240k-stage2.sh` 调用 `src/train/pretrain_unified_navit.py`（配置 `InternVL3_5-2B` + `Qwen3-0.6B`、`Qwen3MoTDecoderLayer`、`--action_representation eef_relative`、`--state_dropout_prob 0.2`、`--vit_dropout_prob 0.2`、`--action_chunk_length 15`）
- **下游数据预处理与微调**：`src/data_postprocessor/{libero,robocasa_human}.py`、`shell/relative-post-libero-full-eef_relative-5shot.sh`、`shell/relative-post-robocasa-full-eef_relative.sh`
- **推理服务端与仿真评测**：`src/inference/{run_server_vla.py,internvl_service.py,internvl_policy.py}` 及 `shell/eval-{libero-relative,robocasa-relative}.sh`、`src/benchmark/{libero,robocasa}/run_*_eval_fast.py`

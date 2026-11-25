# TTP 仓库核查

> 来源归档（repo / BeingBeyond GitHub）

- **URL：** <https://github.com/BeingBeyond/TTP>
- **机构 / 个人：** BeingBeyond（智在无界）
- **仓库核查commit：** `591815c`（84 个文件，含 75 个 `.py`/`.sh`/`.yaml` 源码与配置）
- **开源结论：** **已开源完整预训练、微调、推理服务、评测源码及 Hugging Face 模型权重与 H-Tac 示例子集**（`BeingBeyond/TTP`、`BeingBeyond/H-Tac_Sample`；完整 H-Tac 数据集标注 coming soon）。
- **核对范围：** 已通过 `gh api` 核对 `BeingBeyond/TTP @ 591815c` 完整文件树、`README.md` 并直接阅读 `scripts/train/train_human_tactile.sh` 源码脚本；未实际运行 GPU 训练或真机测试。
- **通读：** 已完成

## 核心源码入口与模块

- **人类触觉预训练与后训练**：`scripts/train/train_human_tactile.sh`（基于 `InternVL3_5-2B` + `Qwen3-0.6B` / `Being-H05-2B` 恢复权重，`USE_MPG=True`、`ACTION_CHUNK_LENGTH=32`、`HISTORY_SIZE=8`、`HISTORY_STRIDE=4`，调用 `BeingH/train/pretrain.py` 与 `BeingH/train/train.py`）；下游仿真训练 `scripts/train/train_{libero_example,robocasa}.sh`
- **数据转换与多模态数据集**：`BeingH/dataset/{lerobot_convert_human_tactile.py,datasets/hmg_dataset.py,datasets/vla_dataset.py}` 及 `configs/posttrain/tactile/tactile.yaml`
- **模型主干与双专家服务**：`BeingH/model/beingvla.py`（理解 + 动作 + 触觉专家与 MPG 门控）、`BeingH/inference/{run_server_TTP.py,real_robot_server.py,real_robot_client.py}`
- **仿真基准评测**：`scripts/eval/eval-{libero-fast,libero-plus,robocasa-multiprocess}.sh` 调用 `BeingH/benchmark/` 下对应评测器

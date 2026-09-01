# VLAct GitHub 仓库

> 来源归档（ingest）

- **项目名称：** VLAct
- **GitHub 地址：** <https://github.com/starVLA/VLAct>
- **许可证：** MIT
- **核心功能：** Qwen3-VL-4B 上的表征中心 VLA **持续预训练**与多 benchmark 下游微调 / 评测 / 真机部署。
- **入库日期：** 2026-09-01

## 仓库结构（README 对齐）

| 路径 | 作用 |
|------|------|
| `starVLA/model/framework/QwenHybrid_xrobot_padding.py` | VLAct 共享 latent → OFT + PI + GR00T 多头 |
| `starVLA/model/framework/{QwenOFT,QwenPI_v4,QwenGR00T}.py` | 各动作头实现 |
| `starVLA/training/train_starvla{,_cotrain}.py` | VLA / VLA+VLM 共训练入口 |
| `scripts/run_scripts/Pretrain/` | 持续预训练（单节点 / Slurm） |
| `scripts/run_scripts/{LIBERO,VLA-Arena,RoboTwin,DOMINO}/` | 下游训练与评测 |
| `examples/{DROID,InternA1,MolmoAct,RoboCoin}/` | 预训练数据清洗与 cache |
| `examples/{LIBERO-plus,VLA-Arena,Robotwin,DOMINO,Robocasa_tabletop}/` | 各 benchmark 环境说明 |
| `deployment/` | 真机 policy server |

## 关键复现路径

1. `conda` + `requirements.txt` + `flash-attn` + `pip install -e .`
2. 持续预训练：`bash scripts/run_scripts/Pretrain/pretrain_qwen3_single_node.sh`
3. 下游示例：`bash scripts/run_scripts/RoboTwin/train_robotwin_qwen3oft.sh`
4. HF 骨干：`huggingface-cli download` → 下游 launcher 设 `pretrained_ckpt`

## 关联 Wiki 页面

- [VLAct 论文实体](../../wiki/entities/paper-vlact.md)
- [StarVLA 方法](../../wiki/methods/star-vla.md)
- [VLA](../../wiki/methods/vla.md)

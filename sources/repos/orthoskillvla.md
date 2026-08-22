# OrthoSkillVLA（Jiaqi-Wangx/OrthoSkillVLA）

- **URL：** <https://github.com/Jiaqi-Wangx/OrthoSkillVLA>
- **论文：** [arXiv:2608.19589](https://arxiv.org/abs/2608.19589)（PRCV 2026）
- **预训练：** <https://huggingface.co/Jiaqi-Wangx/pretrained_xvla>
- **数据：** <https://huggingface.co/datasets/Jiaqi-Wangx/libero_90_xvla>

## 入口

```bash
uv sync --all-groups
CUDA_VISIBLE_DEVICES=0 bash scripts/train_orthoskillvla.sh 1 open_close turn pick_place otp
```

- 仿真评测：`scripts/deploy.py` + `sim_eval/libero/libero_client-skills.py`
- 技能划分：`sim_eval/libero/libero_skills.json`

## 状态（2026-08-22）

**已开源**；训练脚本限定单进程单卡。

## wiki

- [`wiki/entities/paper-orthoskillvla.md`](../../wiki/entities/paper-orthoskillvla.md)

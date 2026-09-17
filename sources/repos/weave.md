# WEAVE

> 来源归档

- **标题：** WEAVE
- **类型：** repo
- **链接：** <https://github.com/xiaohu-art/Weave>
- **论文：** <https://arxiv.org/abs/2609.16683>
- **数据集：** <https://huggingface.co/datasets/appolyn/Weave>
- **项目页：** <https://xiaohu-art.github.io/Weave/>
- **入库日期：** 2026-09-17
- **一句话说明：** G1+Inspire HOI 跟踪：Isaac Lab / RSL-RL PPO，Hydra `configs/track/train.yaml`，九物体联合训练。
- **代码：** **已开源**
- **沉淀到 wiki：** [`paper-weave`](../../wiki/entities/paper-weave.md)
- **交叉归档：** [`weave_arxiv_2609_16683.md`](../papers/weave_arxiv_2609_16683.md)、[`weave-xiaohu-art.md`](../sites/weave-xiaohu-art.md)

## 核心入口

| 路径 | 说明 |
|------|------|
| `install.sh` | Isaac Sim + Isaac Lab + 依赖安装 |
| `scripts/rsl_rl/train.py` | PPO 训练（默认 headless） |
| `configs/track/train.yaml` | 九物体联合训练默认配置 |
| `configs/track/eval.yaml` / `play.yaml` | 评测与可视化 |
| `scripts/list_envs.py` | 安装后环境自检 |

## 复现要点

```bash
python scripts/rsl_rl/train.py --task=G1-Inspire-HOI-v0 \
    --config-dir ./configs/track --config-name train
```

- 默认任务 **G1-Inspire-HOI-v0**；日志 `logs/rsl_rl/g1_inspire_hoi/<timestamp>_<run_name>/`。
- README 另提及 `g1_hoi_learning` 克隆路径；截至 **2026-09-17** 主实现已在 **本仓**（`train.py` / `install.sh` 可访问）；`g1_hoi_learning` URL **404**。

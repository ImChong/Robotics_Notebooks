# A-MDM: Interactive Character Control with Auto-Regressive Motion Diffusion Models — 官方源码

- **源码：** https://github.com/Yi-Shi94/AMDM
- **论文：** https://arxiv.org/abs/2306.00416v4
- **项目页：** https://yi-shi94.github.io/amdm_page/
- **开放情况：** 官方仓库可访问；具体训练环境、预训练权重与依赖需要按仓库 README 单独配置。
- **核对范围：** GitHub README 与明确标注的源码/命令入口；尚未运行训练、评测或真机部署测试。
- **通读：** 未完成

## 运行入口与限制

官方 README 给出的训练入口 `python run_base.py --arg_file args/amdm_DATASET_train.txt`，推理 `python run_env.py --arg_file args/RP_amdm_DATASET.txt`，inpainting 使用 `args/PI_amdm_DATASET.txt`。

## 对 wiki 的映射

- [A-MDM：自回归扩散角色交互控制](../../wiki/entities/paper-a-mdm-interactive-character-control.md)

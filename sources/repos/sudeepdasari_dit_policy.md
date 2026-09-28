# dit-policy（SudeepDasari/dit-policy）

- **URL：** <https://github.com/SudeepDasari/dit-policy>
- **项目页：** <https://dit-policy.github.io/>
- **论文：** [robotic_dit_ingredients_arxiv_2410_10088.md](../papers/robotic_dit_ingredients_arxiv_2410_10088.md)
- **License：** MIT
- **入库日期：** 2026-09-28

## 一句话说明

DiT-Block Policy 官方实现：`finetune.py` 训练 **adaLN 扩散 Transformer**（`agent=diffusion`）与 **U-Net Diffusion Policy** baseline；数据需转为 robobuf；含 ALOHA/DROID 部署 eval 脚本入口。

## 关键入口（README）

| 路径 / 命令 | 用途 |
|-------------|------|
| `finetune.py` … `agent=diffusion` `ac_chunk=100` | DiT-Block Policy 训练 |
| `finetune.py` … `agent=diffusion_unet` | U-Net DP baseline |
| `eval_scripts/` | 真机 / 仿真评测 |
| `env.yml` | Conda 环境 |

## 交叉链接

- [paper-robotic-dit-ingredients-dit-block-policy](../../wiki/entities/paper-robotic-dit-ingredients-dit-block-policy.md)
- [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md)

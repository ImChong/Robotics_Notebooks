# dit-policy（SudeepDasari/dit-policy）

- **URL：** <https://github.com/SudeepDasari/dit-policy>
- **项目页：** <https://dit-policy.github.io/>
- **论文：** [robotic_dit_ingredients_arxiv_2410_10088.md](../papers/robotic_dit_ingredients_arxiv_2410_10088.md)
- **License：** MIT
- **入库日期：** 2026-09-28
- **最近复核：** 2026-11-19
- **核对范围：** `README.md`、`finetune.py`、`eval_scripts/eval_aloha.py`、`experiments/finetune.yaml`、`experiments/agent/diffusion.yaml`、`data4robotics/agent.py`、`data4robotics/models/diffusion.py`
- **通读：** 已完成

## 一句话说明

DiT-Block Policy 官方实现：以 Hydra 驱动 `finetune.py` 训练 **adaLN-Zero 扩散 Transformer**（`agent=diffusion`）与 U-Net 等基线，并提供带时间集成与 `torch.compile` 加速的真机/ALOHA 部署脚本。

## 关键入口与模块（源码核对）

| 路径 / 符号 | 用途 | 关键实现细节 |
|-------------|------|--------------|
| `finetune.py` (`bc_finetune`) | 主训练/微调入口 | `@hydra.main` 加载 `experiments/finetune.yaml`，实例化 `trainer`、`task`、`agent`（如 `DiffusionTransformerAgent`），执行训练循环 |
| `experiments/agent/diffusion.yaml` | DiT 策略默认配置 | `num_blocks: 6`、`hidden_dim: 512`、`nhead: 8`、`features: resnet_gn_nopool`、`train_diffusion_steps: 100`、`eval_diffusion_steps: 8`（DDIM `squaredcos_cap_v2`） |
| `data4robotics/agent.py` (`BaseAgent`) | 策略基类与本体编码 | 处理多视角视觉特征提取并对低维本体状态施加 `nn.Dropout(p=0.2)` 防捷径学习 |
| `data4robotics/models/diffusion.py` (`DiffusionTransformerAgent`, `_DiTNoiseNet`) | DiT-Block 去噪网络 | `_SelfAttnEncoder` 融合观测 token；`_DiTDecoder` 内含 `_ShiftScaleMod` 与零初始化残差 `_ZeroScaleMod`（adaLN-Zero）；推理先单次运行 `forward_enc` 缓存编码特征，再循环执行 `forward_dec` 完成 DDIM 去噪 |
| `eval_scripts/eval_aloha.py` (`Policy`) | ALOHA 真机闭环评测 | 使用 `torch.compile(agent.eval().cuda().get_actions)` 加速推理；支持 `_forward_ensemble`（`pred_horizon=48`、EMA `gamma=0.85`）与 `_forward_chunked` |

## 交叉链接

- [paper-robotic-dit-ingredients-dit-block-policy](../../wiki/entities/paper-robotic-dit-ingredients-dit-block-policy.md)
- [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md)

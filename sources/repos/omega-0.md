# gentlefress/OMEGA-0：官方训练与部署代码

- **类型：** repo / humanoid / world-action-model / loco-manipulation
- **仓库：** <https://github.com/gentlefress/Omega-0>
- **项目页：** <https://gentlefress.github.io/OMEGA-0_page/>（[归档](../sites/omega0-github-io.md)）
- **论文：** <https://arxiv.org/abs/2608.06375>（[摘录](../papers/omega0_arxiv_2608_06375.md)）
- **数据集：** <https://huggingface.co/datasets/keycharon/omega-HOME>（[归档](../datasets/omega-home.md)）
- **许可证：** MIT（仓库注明第三方代码/文件仍依其各自许可证）
- **核查日期：** 2026-10-03
- **一句话说明：** ω-0 官方 G1 代码仓，包含动作 token 预训练、WAM 微调、推理、Pico 遥操作/数据采集、episode 录制及真机部署入口。

## 已开放内容与边界

README 给出 Unitree G1 + Inspire 手 + ZED Mini ego 相机配置，基于 SONIC 低层控制，文档覆盖环境安装、数据采集、推理服务、机器人客户端和两阶段训练配方。仓库标注 MIT。

**权重仍未发布。** README 的 TODO 仍列出发布预训练 checkpoints；实际训练也需要 Qwen3-VL、FAST tokenizer、T5、V-JEPA、Wan VAE 和 predictor 初始化等外部模型资产。仓库公开了代码与流程，不代表提供了开箱可复现的预训练权重或完整数据包。

## 主要入口

| 内容 | 仓库位置 / 说明 |
|------|----------------|
| 全身动作 token 预训练 | `src/configs/vlm_pretrain.yaml`；FAST tokenizer + Qwen3-VL |
| WAM 微调 | `src/configs/finetune.yaml`；动作预测与未来视觉表征学习 |
| 推理服务 | `src/configs/serve.yaml` |
| G1 采集与部署 | `real/configs/collect.yaml`、`real/configs/deploy.yaml` |
| 第三方 SONIC 部署后端 | `thirdparty/gear_sonic_deploy` |

## 对 wiki 的映射

- [ω-0 论文节点](../../wiki/entities/paper-omega-0.md)
- [项目主页归档](../sites/omega0-github-io.md)
- [ω-HOME 数据集归档](../datasets/omega-home.md)

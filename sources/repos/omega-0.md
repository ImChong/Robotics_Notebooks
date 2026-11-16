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

## 固定版本实现复核（2026-11-13）

- **版本：** [2ca35a3ff9363c54c9c206ca101cdf58028824cb](https://github.com/gentlefress/Omega-0/tree/2ca35a3ff9363c54c9c206ca101cdf58028824cb)
- **核对范围：** 根README、inference/run.py与model.py全文；Service.reset/predict/handle，models/wam.py的_encode_inputs与sample完整函数，configs/finetune.yaml全文。
- **通读：** 已完成
- 当前TODO检查点已勾选并链接keycharon/Omega-zero，不再待发布。训练/部署/采集开放，但主流程依赖本地Qwen/T5/VJEPA/Wan/SONIC资产。
- HTTP /act走Service→WAMInference.predict→preprocess→model.sample→反归一化。请求带session/observation/action spec；reset使旧响应失效。RTC从上一未执行动作区间取前缀填入初始噪声，idle或view改变重置。
- 现成finetune recipe为30步66D、predictor-only、无state、normalization none；论文附录H25、47Dstate与混合归一化不同，不把默认脚本当Full复现。训练导出metadata决定推理预处理，必须一致。静态核读，未跑GPU或机器人。

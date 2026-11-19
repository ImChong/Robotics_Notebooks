# Improved Baselines with Representation Autoencoders（RAEv2）：改进表征自编码器基线

> 论文一手资料核读记录：只归纳原文、公式、关键实验与复现约束，不整篇转存论文。

- **标题：** Improved Baselines with Representation Autoencoders / 改进表征自编码器基线
- **类型：** paper / image generation / representation autoencoder / flow matching / navigation world model
- **作者：** Jaskirat Singh、Boyang Zheng、Zongze Wu、Richard Zhang、Eli Shechtman、Saining Xie
- **机构：** Adobe Research、澳大利亚国立大学（Australian National University, ANU）、纽约大学（New York University, NYU）
- **论文：** [arXiv:2605.18324v2](https://arxiv.org/abs/2605.18324v2)，v1 2026-05-18；v2 2026-06-15
- **HTML：** https://arxiv.org/html/2605.18324v2
- **PDF：** https://arxiv.org/pdf/2605.18324v2
- **项目：** https://raev2.github.io/
- **代码：** https://github.com/nanovisionx/RAEv2 （项目页的 Code 官方链接）
- **模型：** https://huggingface.co/nyu-visionx/RAEv2-models
- **数据集：** https://huggingface.co/datasets/nanovisionx/RAEv2-data
- **核对日期：** 2026-11-19
- **核对范围：** 通读 arXiv v2 HTML §1–§6、§2.1–2.3 的层聚合/REPA/自引导机制，`3 的 K 和引导消融、§4 的 T2I 与导航世界模型，附录 A 实现配置、附录 C 指标表、附录 E 局限；核对项目页和官方 README、Stage1/Stage2 训练命令以及 src/train_stage1.py、src/stage1/engine.py、src/stage1/rae.py、src/train.py、src/stage2/engine.py、configs/stage2/training/nwm-dinov3l-k7.yaml。
- **通读：** 已完成

## 一句话说明

RAEv2 保留冻结预训练视觉编码器的语义 latent，利用末 K 层特征求和提高可重建性，用 REPA 正则化扩散中间特征的空间结构，再把 REPA 早层预测头复用为内部引导，避免另训一套弱模型。

## 三个可核对的机制

1. **Generalized RAE（§2.1）：** 最后 K 层 patch tokens 求和，式 (1) 中 x = sum(z_l)，形状始终为 N×d；K=1 退化为原始 RAE。不需微调视觉编码器。附录 C 对比 MLS（多层求和）与 MLR（随机投影），相同 K 下 MLS 的 Stage2 gFID 更好。
2. **RAE + REPA（§2.2）：** 在 27 种编码器上的分析显示两者互补：RAE 利用 encoder 的全局语义，REPA 改善 DiT 中间 token 的局部空间相似性；可把同一预训练表征同时当输入 latent 与监督 target。
3. **REPA Self-Guidance（§2.3）：** REPA 的早层 projection head 是 clean latent 的 x-prediction，满层 DiT 输出也重参数化为 x-prediction；两个预测可在同一潜空间做 guidance，无需为 AutoGuidance 额外训练模型，也无需标准 CFG 第二次前向。此结论不能扩展为所有任务、所有引导强度下都没有额外计算。

## 核心数字（作者报告，非本站复现）

| 设定 | 比较对象 | RAEv2 | 证据 |
|---|---|---|---|
| ImageNet-256，K=7，80 epochs，有引导 | 原始 RAE 800 epochs gFID 1.13 | gFID **1.06** | 表 7 |
| 收敛效率 EP_FID@2（无引导 gFID 达 2 的 epoch） | RAE **177** | **35** | 表 7 |
| ImageNet-256，多表征距离 FDr6 | 原始 RAE 800 epochs **3.26** | 80 epochs **2.17** | 表 7 |
| ImageNet-256，K=7，80 epochs，引导消融 | 无引导 1.65 / CFG 1.49 / AG 1.14 | REPA guidance **1.06** | 表 16 |
| T2I（GenEval，微调后） | Flux-VAE **78.3** / RAE **81.5** | **82.7** | 表 8 |
| RECON 导航视频，至 16 秒 rollout 的 FVD | RAE **312.01** / NWM **200.97** | **105.61** | 表 9/21 |

**注意：** “超过 10× 收敛加速”是论文报告的整体训练收敛/预算对照；表 7 的 EP_FID@2 比值为 177/35≈5.1，不能将两种口径混成同一个数字。

## 导航世界模型与机器人工程边界

- 任务是 **动作条件的下一帧 / 未来帧生成**，不是直接输出电机控制指令。
- §4.2：4 帧历史上下文 × 每帧 16×16 tokens = 1024 个图像条件 tokens；另有 4 个动作 tokens，动作表示为本体坐标增量 (Δx, Δy, Δψ)，及 rollout 时间 token。基于 RECON 4 FPS 自回归生成，按 1/2/4/8/16 秒 horizon 验证。
- rollout 读取的是 **ground-truth action**；不能据 FVD 改善推断真机闭环导航成功率或物理一致性。
- 附录 E：仅比较多层求和与随机投影两种简单聚合；表征仍依赖在有限视觉编码器集合中的经验选择，没有证明联合端到端优化或跨所有物理场景的最优性。

## 官方实现对应关系

- Stage1：`src/train_stage1.py` → `src/stage1/engine.py` → 冻结 `stage1.RAE` 的 encoder、训练 decoder（L1 / LPIPS / GAN），输出 checkpoint，再用 `scripts/stage1/extract_decoder.py` 和 `scripts/stage1/compute_encoder_stats.py` 处理 Stage2 所需权重与统计。
- Stage2：`src/train.py` → `prepare_unified_dataloader` → 冻结 `RAE.encode` / 单独 REPA target encoder → `src/stage2/engine.py::train_one_epoch` 中的 `transport.training_losses`（diffusion/flow loss + REPA loss）→ EMA、采样和评测。
- 官方 README 分别公开 ImageNet、T2I、NWM 配置、权重和数据下载；未在本次操作下载完整权重或实测训练指标。

## 对 wiki 的映射

- [RAEv2 独立论文详情](../../wiki/entities/paper-raev2-improved-representation-autoencoders.md)
- [THAW-VLA：表征对齐用于机器人策略](../../wiki/entities/paper-thaw-vla.md)
- [模型架构 Stage2：感知表征接口](../../roadmap/depth-model-architecture.md)

## 参考来源

- [arXiv v2 HTML 全文](https://arxiv.org/html/2605.18324v2)
- [项目主页](https://raev2.github.io/)
- [官方 PyTorch 代码](https://github.com/nanovisionx/RAEv2)

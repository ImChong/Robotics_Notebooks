# sonic-agibot-x2.github.io/sonic-transfer（冻结 GEAR-SONIC → AgiBot X2 跨具身迁移项目页）

> 来源归档（ingest · 项目页）

- **标题：** Cross-Embodiment Transfer of a Frozen Humanoid Whole-Body Controller via Analytic Codec and LoRA Adapters
- **类型：** site / project-page
- **官方入口：** <https://sonic-agibot-x2.github.io/sonic-transfer/>
- **站点仓：** <https://github.com/sonic-agibot-x2/sonic-agibot-x2.github.io>（子目录 `sonic-transfer/`）
- **draft PDF：** <https://sonic-agibot-x2.github.io/sonic-transfer/static/pdfs/paper.pdf>
- **代码：** <https://github.com/meetsitaram/sonic-x2> — 归档见 [`sources/repos/sonic-x2.md`](../repos/sonic-x2.md)
- **权重：** <https://huggingface.co/tinkerbuggy/sonic-x2>
- **入库日期：** 2026-08-17
- **一句话说明：** 把公开发布、**全程冻结**的 GEAR-SONIC（Unitree G1）全身跟踪控制器，用闭式关节 codec + 动力学解码器上的 LoRA（约 0.25% 参数）迁到 AgiBot X2 Ultra；OOD（PHUMA）成功率 **69.0% vs 原生 incumbent 59.0%**。
- **开源状态（2026-08-17 核查）：** **部分开源、推理可运行**。项目页 Code 指向 `meetsitaram/sonic-x2` 的 `./play_v2.sh`（MuJoCo ONNX 回放 + transfer checkpoint + `.phi.json` codec sidecar）；HF 有模型卡。仓内 **无 SPDX LICENSE**。LoRA **训练脚本不在 play bundle**；完整部署栈另见 [`meetsitaram/GR00T-WholeBodyControl-X2-review`](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2-review)（本条未深挖）。截至入库日 **无 arXiv 编号**。

## 页面公开信息

| 资源 | URL |
|------|-----|
| 项目首页 | <https://sonic-agibot-x2.github.io/sonic-transfer/> |
| draft PDF | <https://sonic-agibot-x2.github.io/sonic-transfer/static/pdfs/paper.pdf> |
| 代码（play） | <https://github.com/meetsitaram/sonic-x2> |
| 权重 | <https://huggingface.co/tinkerbuggy/sonic-x2> |
| 同源 incumbent 工程页 | <https://sonic-agibot-x2.github.io/>（*Porting NVIDIA Sonic to the AgiBot X2 Ultra*） |
| Companion：冻结规划器 | <https://sonic-agibot-x2.github.io/kplanner/> |

作者行：Sitarama Chekuri · Claude Fable 5；标注 **Anthropic（AI co-author）** · draft preprint, 2026。

页首 pill：Paper（draft PDF）· Models · Code（`./play_v2.sh`）· Companion frozen planner · SONIC-X2 port。

## 公开信息要点（截至入库日）

- **冻结边界：** 三个编码器、FSQ token 瓶颈、运动先验与运动学解码器 **bit-identical**；只训动力学解码器上的 LoRA（约平台参数的 **0.25%**）。
- **对齐：** 近亲骨架（G1 ↔ X2 Ultra 关节一一对应）→ **闭式 per-joint affine codec**，不是 Any2Any 那种可学习两级对齐。
- **算力：** 一夜 8-GPU 节点，约平台 cited 训练算力的 **2%**（文中分母取 Any2Any 引用的 ~9k GPU-h，而非 SONIC 全文 21k）。
- **主结果（IsaacLab 严格门；成功% / mean mm / p95 mm）：**

| model | novel500 (in-dist) | hard300v3 (tail) | PHUMA (OOD) | PHUMA survival |
|-------|--------------------|------------------|-------------|----------------|
| incumbent（原生 X2） | 96.4 / 33.6 / 43.8 | 70.0 / 40.6 / 56.0 | 59.0 / 42.6 / 67.4 | 87.4 |
| transfer（breadth only） | 95.6 / 33.7 / 42.9 | 71.0 / 40.0 / 56.4 | 61.6 / 42.9 / 60.9 | 89.4 |
| transfer（selected） | 96.2 / 33.1 / 42.9 | 72.3 / 39.9 / 55.8 | **69.0** / 41.7 / 60.6 | **90.7** |

- **演示：** incumbent vs codec-only zero-shot vs adapted（relaxed walk / Gangnam dance）；手腕固定在 deploy-default。
- **与 Any2Any 的页内定位：** Any2Any = 低成本 **parity**（跨形态差较大的人形对）；本页 = 近亲骨架上的冻结平台 **reversal**。

## 为何值得保留

- 把「已有 Gear-SONIC 专家怎么搬家」从 [Any2Any](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md) 的 **可学习对齐 + 多模块 LoRA** 推进到 **闭式 codec + 单解码器 LoRA + 更严冻结**，并给出 **OOD 反超原生 tracker** 的数字。
- 工程可跑：官方 play bundle 默认就是 transfer ONNX，不是只放视频。
- 明确写出 **embodiment cost floor** 与 **冻结 latent 信息瓶颈**（腕关节 reward-invisible），可直接进选型误区。

## 关联资料

- 论文摘录：[`sources/papers/sonic_transfer_frozen_wbc_codec_lora.md`](../papers/sonic_transfer_frozen_wbc_codec_lora.md)
- 代码仓：[`sources/repos/sonic-x2.md`](../repos/sonic-x2.md)
- 升格：[`wiki/entities/paper-sonic-transfer.md`](../../wiki/entities/paper-sonic-transfer.md)

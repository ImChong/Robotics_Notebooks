# The Ingredients for Robotic Diffusion Transformers（arXiv:2410.10088）

> 论文来源归档（ingest）

- **标题：** The Ingredients for Robotic Diffusion Transformers
- **作者：** Sudeep Dasari, Oier Mees, Sebastian Zhao, Mohan Kumar Srirama, Sergey Levine（CMU / UC Berkeley）
- **类型：** paper / diffusion-policy / transformer / manipulation / bimanual
- **arXiv：** <https://arxiv.org/abs/2410.10088> · PDF：<https://arxiv.org/pdf/2410.10088.pdf>
- **项目页：** <https://dit-policy.github.io/> — [`sources/sites/dit-policy-github-io.md`](../sites/dit-policy-github-io.md)
- **代码：** <https://github.com/SudeepDasari/dit-policy> — [`sources/repos/sudeepdasari_dit_policy.md`](../repos/sudeepdasari_dit_policy.md)
- **数据：** [BiPlay](https://huggingface.co/datasets/oier-mees/BiPlay)（~7k play clips，9.7h 双臂 ALOHA，覆盖 326 场景 / 200+ 任务）
- **入库日期：** 2026-09-28
- **复核日期：** 2026-11-19
- **核对范围：** 通读 arXiv:2410.10088 全文文本（摘要、引言、相关工作、方法架构、实验与消融 Table I–V 及结论）
- **通读：** 已完成
- **一句话说明：** 系统拆解**扩散 Transformer 策略**的关键设计配方——**分相机独立 ResNet + FiLM 语言调制 + 本体 dropout + adaLN-Zero 解码器**构成 **DiT-Block Policy**，解决原生 cross-attention Transformer 难训练问题。

## 核心摘录（面向 wiki 编译）

### 1) 为何 U-Net DP 之外需要「配方」

- **要点：** 论文指出 Diffusion Policy 原始 cross-attention Transformer 噪声网络极难调参且训练不稳定；社区多退回 1D U-Net，但其时序平滑归纳偏置限制了在高容量预训练和多模态数据上的扩展。
- **对 wiki 的映射：** [`wiki/entities/paper-diffusion-policy.md`](../../wiki/entities/paper-diffusion-policy.md)、[`roadmap/depth-robotics-diffusion-dit-flow.md`](../../roadmap/depth-robotics-diffusion-dit-flow.md)

### 2) adaLN-Zero 稳定扩散 Transformer 策略（Table III）

- **要点：** 在去噪解码器中用 **adaLN-Zero** 替代标准 cross-attention / in-context 拼接，将观测条件 token 与扩散时间步映射为 LayerNorm 的 scale/shift 及零初始化残差门控。Table III（Pick Place / Pen Uncap）表明：10 步 DDIM 下 Cross Attn. 成功率为 `0% / 0%`（增至 100 步才达 `38% / 70%`），In Context 100 步仍为 `0% / 0%`，去掉零初始化（No Zero-Init）降为 `38% / 80%`，而完整 DiT-Block 在 10 步即可达 `50% / 100%`。
- **对 wiki 的映射：** [`wiki/entities/paper-dit-scalable-diffusion-transformers.md`](../../wiki/entities/paper-dit-scalable-diffusion-transformers.md)

### 3) 观测 token 化：分相机 ResNet + 本体 dropout（Table IV）

- **要点：** 各相机使用**独立 ResNet 编码器**（论文正文写 ResNet-26，项目页展示为 ResNet-18，代码默认支持 `resnet_gn` / `resnet_gn_nopool`）；DistilBERT 语言特征通过 **FiLM** 调制视觉层；低维本体感知加入 `dropout (p=0.2)` 防止策略依赖本体捷径；视觉、语言与本体 token 经 Octo 式非因果注意力编码器融合。Table IV 消融显示，用三层卷积 stem 替代 ResNet（No ResNet，85M 参数与 150M 参数缩放版本）在 ALOHA Pick Place 上均跌至 `0%`，Pen Uncap 仅 `13% / 20%`，远低于 115M DiT-Block 的 `50% / 100%`。
- **对 wiki 的映射：** [`wiki/methods/action-chunking.md`](../../wiki/methods/action-chunking.md)

### 4) 多任务评测与跨本体验证（Table II、Table V、Fig. 5）

- **Table II（双臂 ALOHA，含 BiPlay / 不含 BiPlay）：** 含 BiPlay 时平均成功率 DiT-Block **60% ± 9%** vs D.P. U-Net **42% ± 9%** vs ACT **33% ± 8%** vs D.P. Transformer **0% ± 0%**（Pick Place `50%`, Pen Uncap `100%`, Sushi Cut `29%`）；不含 BiPlay 时 DiT-Block 仍为 **51% ± 9%**（vs ACT `37%`, D.P. U-Net `30%`, D.P. Transformer `0%`）。
- **单臂 Franka FR3（Fig. 5，笛卡尔速度空间，单第三视角）：** Toasting 与 Wiping 两项任务平均超越 ACT 20%、超越 D.P. U-Net 35%。
- **robomimic 仿真（Table V）：** DiT-Block（无逐任务特调）4 项平均 **88.5%**（Lift 100%, Can 98%, Square 84%, Tool Hang 72%），接近重度调参的 D.P. Transformer 基准 **94%**（100%/100%/100%/76%），同时后者在真机 ALOHA 上为 **0%**，凸显真机与仿真评测差距。

### 5) Action Chunk 与调度超参

- **要点：** 训练预测 `H=100`（代码评估脚本 `eval_aloha.py` 支持 `pred_horizon=48` 与 EMA 时间集成）；训练使用 DDPM `k=100` 步与 `squaredcos_cap_v2` 余弦调度；论文报告推理用 **10 步**确定性 DDIM 采样（公开代码 `experiments/agent/diffusion.yaml` 默认设 `eval_diffusion_steps: 8`）。
- **对 wiki 的映射：** [`wiki/concepts/receding-horizon-policy-execution.md`](../../wiki/concepts/receding-horizon-policy-execution.md)

## 开源状态（步骤 2.5，2026-11-19 复核）

| 资源 | 状态 |
|------|------|
| [SudeepDasari/dit-policy](https://github.com/SudeepDasari/dit-policy) | **已开源**（MIT；含 `finetune.py`、`agent=diffusion` DiT-Block、`agent=diffusion_unet`、ALOHA 真机评测 `eval_scripts/eval_aloha.py`） |
| [BiPlay](https://huggingface.co/datasets/oier-mees/BiPlay) | **已发布**（HF Dataset） |
| 预训练视觉权重 | 通过 `download_features.sh` / CMU data4robotics 发布获取 |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 升格实体：[`wiki/entities/paper-robotic-dit-ingredients-dit-block-policy.md`](../../wiki/entities/paper-robotic-dit-ingredients-dit-block-policy.md)

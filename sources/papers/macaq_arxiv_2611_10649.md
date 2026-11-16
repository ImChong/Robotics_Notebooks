# MACaQ：面向低精度权重与激活的快速、模型无关校准量化

- **英文标题：** MACaQ: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations
- **中文标题：** MACaQ：面向低精度权重与激活的快速、模型无关校准量化
- **作者：** Lorenz K. Muller、Philippe Bich、Chiara Boretti、Hyun-Min Chang、Jiawei Zhuang、Lukas Cavigelli
- **机构：** 华为计算系统实验室（Huawei Computing Systems Lab, Switzerland）
- **论文：** https://arxiv.org/abs/2611.10649v2
- **HTML：** https://arxiv.org/html/2611.10649v2
- **PDF：** https://arxiv.org/pdf/2611.10649v2
- **首次提交：** 2026-11-07；**v2 修订：** 2026-11-11
- **代码：** https://github.com/huawei-csl/MACaQ
- **官方交互介绍：** https://huawei-csl.com/blog/macaq/
- **模型集合：** https://huggingface.co/collections/huawei-csl/macaq
- **入库日期：** 2026-11-16
- **核对范围：** arXiv:2611.10649v2 HTML 全文：Abstract、Introduction、Method（问题定义、Online Scaling、Online Micro-Rotation、Hessian-Weighted Exponent Selection、Flip-Refinement、Online Cost、Quantization Procedure、Implementation）、Experiments（Table 1–5、Fig. 1–4）、Related Work、Limitations、Conclusion、Appendix 0.A / 0.B / 0.C（吞吐表、Pareto 数据、灵敏度、格式一致性、向量量化扩展）；交叉核对官方交互介绍、GitHub README 与 vLLM/量化脚本。
- **通读：** 已完成
- **沉淀到 wiki：** [MACaQ 论文详情](../../wiki/entities/paper-macaq.md)

## 研究问题与核心创新

MXFP4 W4A4 训练后量化（PTQ）同时压缩权重与激活，但若为了取消运行时变换而将基变换折叠到相邻层，会增加架构相关改造和跨层耦合。MACaQ **逐线性层独立**校准：输入侧在线通道缩放和 32 维块 Hadamard 旋转，权重侧离线吸收逆向对应变换，不需修改相邻归一化/残差分支。

- **块格式：** MXFP4 每 32 个值共享 E8M0 指数尺度，每个值为 E2M1 4-bit 码。不能把 W4A4 误写成模型所有层均为 4 bit：正文 §3.2 对 MoE 仅量化专家层，且不量化 embeddings/lm-head。
- **逐层目标（Eq.1–2）：** 最小化 ||XWᵀ − XŴᵀ||²_F；H=XᵀX 由代表性校准输入累积，而不是端到端任务奖励。
- **等价变换（Eq.11）：** x'=(x⊙t)B、W'=(W⊘t)B；B 为块正交 Hadamard，匹配 32 元素 MXFP4 微缩放块。
- **选择指数（Eq.12）：** 对每个权重行×量化块考察基础指数的 +1、0、−1，按 Hessian 对角加权重建误差筛选，尺度固定后再 refine。
- **Flip-Refinement（Algorithm 1，Eq.13–16）：** 从 RTN 初始码出发，利用 H 加权的局部二次目标，贪心地把单个 FP4 码改至相邻值；每轮每个输出行至多接受一个正收益 flip，活跃行提前退出。
- **逐层流式（Eq.19）：** forward hook 累积每层 H；每次加载当前层到 GPU 校准和量化、替换成打包算子，其余权重留在主机内存；与模型整体需要完全驻留 GPU 不同。
- **在线算量（Eq.17–18）：** 块长 32 时理论运算量比例约 3/m（m 为输出通道数）；**这是算术量估计，不是壁钟时延预测**，真实表现取决于融合 kernel 与内存。

## 论文实验：明确评测口径

| 原文位置 | 设置 | 结果 |
|---|---|---|
| Table 1 | Llama-3-8B，MXFP4 W4A4，WikiText-2 / C4 PPL，越低越好 | MACaQ 6.90 / 11.01；MR-GPTQ 7.13 / 11.37；DuQuant++(GPTQ) 7.17 / 11.66；FP16 6.14 / 9.46 |
| Fig.2 / Table 7 | Llama-3-8B，量化时间与 PPL Pareto | 相同最优级 PPL 时，作者报告约 11× 更短量化时间（取决于校准 token 和 flip 预算） |
| §3.2 | Qwen3.5-122B-A10B，单张 96 GB GPU，32 flips、512K 校准 token | 量化耗时 64 分钟，峰值 GPU 显存 32.5 GB，量化模型使用约 74 GB；PPL 5.14 vs FP16 5.06 |
| Fig.3 / Appendix Table 6 | Qwen3-32B，vLLM，batch 1、context 512 | 在线缩放+旋转相比无该变换的 MXFP4 增加约 2.3% 端到端开销；decode 69.8 vs BF16 22.8 tok/s（约 3.06×） |
| Table 3 | FLUX.2-klein-4B，800 个 GenEval2 prompts，DiT-only W4A4 | MACaQ PSNR 15.04、LPIPS 0.218、峰值 VRAM 13.1 GB；官方 NVFP4 14.85、0.235、13.3 GB，**格式 bit/weight 不同** |
| Table 5 | Llama-3-8B，512 flips、200K token，消融 | RTN 8.28 → 指数搜索 8.09 → 旋转 7.65 → 缩放 7.08 → Flip 6.87（Wiki2 PPL） |

## 原文局限与工程边界

- §5 Limitations：严格层独立不支持跨层误差补偿；0.6B 小模型的在线变换开销约 **8%**；MoE 实验只量化专家层。
- 论文没有机器人 VLA/动作头、真实机械臂、控制周期或真机成功率实验；用于机器人部署仅能作为**待验证的通用 PTQ 候选**。机器人闭环量化敏感性要另参考 VLAQuantBench 等任务级证据。
- 官方交互页的 llama.cpp、Diffusers、PyTorch、vLLM 与 HF 示例不应等同于这份 v2 论文全部正式评测；README 另列扩展 MACaQ-R 与 GGUF 格式，须单独标识。
- 未在本环境实际运行量化、推理或硬件测速；论文数字为作者报告。

## 对 wiki 的映射

- [MACaQ 独立论文实体](../../wiki/entities/paper-macaq.md)
- [模型架构纵深路线 Stage 5](../../roadmap/depth-model-architecture.md)
- [GGUF 量化内幕](../../wiki/entities/gguf-quant-internals.md)
- [VLAQuantBench](../../wiki/entities/paper-vlaquantbench.md)（机器人闭环验收对照）

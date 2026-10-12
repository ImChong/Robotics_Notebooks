# ESP 论文来源归档

## 元数据

- **题目：** ESP: Energy-Score Policy for One-Step Multimodal Action Generation
- **方法：** Energy-Score Policy（ESP）
- **作者：** Lilika Makabe, Heecheol Kim, Yasuyuki Matsushita
- **单位：** arXiv 页与项目页均未列出（项目页仅链到 Matsushita 的 Microsoft Research 个人页；机构归属未核实）
- **预印本：** https://arxiv.org/abs/2610.07696
- **HTML：** https://arxiv.org/html/2610.07696v1
- **版本：** v1，2026-10-06，cs.RO，CC BY 4.0
- **项目页：** https://lilika-makabe.github.io/energy-score-policy-site/ （归档见 [esp-energy-score-policy-project.md](../sites/esp-energy-score-policy-project.md)）
- **代码：** 截至 2026-10-12 项目页与论文均未提供代码链接。

## 论文要解决的问题

扩散 / 流匹配动作头能表示同一场景下多种合理动作序列，但每个动作块需要多次网络评估，拖慢闭环控制；MSE 回归一次前向就能输出，却收敛到条件均值，把多个模态“抹平”。ESP 想要一次评估且保持多模态。

## 方法与训练信息

1. 无教师（teacher-free）：直接把策略上下文与噪声 z ~ N(0, I) 映射为动作块，推理 NFE = 1。
2. 训练损失用能量分数（β = 1）：ES(Q, y) = E‖x−y‖ − ½E‖x−x′‖，它是严格正当（strictly proper）评分规则，期望只在真实目标分布处取最小；有限样本估计用每个训练样本 K = 8 个候选。
3. π0.5 / LIBERO：冻结视觉语言骨干，把流匹配动作头换成一步 ESP 头（时间输入固定 t = 1，随机初始化），批 32，AdamW，EMA 0.999，60k 步，3 个 seed，预测 10 步执行 5 步。
4. GR00T-N1.7-3B / 真机：冻结 VLM 骨干，只微调动作头，批 8，lr 2.5e-5，预测并执行 8 步。
5. 基线：XM（K 个候选取最小，NFE=1）、原生流匹配（π0.5 为 NFE=10，GR00T 为 NFE=4）、MSE。

## 论文报告的结果

- TwoBranch 双模态回归 Match rate（单 seed）：MSE 0.7%，一步流 4.2%，XM 91.9%，ESP 96.2%，10 步流 97.1%。
- LIBERO 平均成功率（π0.5，3 seed）：ESP 0.918 ± 0.014，XM 0.866，Flow(NFE 10) 0.891；Long10 上 ESP 0.813，XM 0.672，Flow 0.802。
- 延迟（A100，batch 1）：动作专家 ESP 3.35 ms vs Flow 28.39 ms（约 8.47×）；缓存端到端 4.46 vs 29.14 ms（6.53×）；未缓存端到端 59.01 vs 82.54 ms（1.40×）。
- 动作专家加宽（FFN 73728，总 7.204B 参数）：平均 0.937，延迟 21.15 ms。
- 多模态消融（6 个 LIBERO-10 任务 × 32 rollout）：成功 ESP 162、XM 127、Flow 155；RMS 对称 Chamfer 离散度 ESP 2.122 cm，XM 1.965 cm，Flow 2.280 cm。
- 真机（Franka Research 3，GR00T，每项 15 次）：Drawer Closing 12/15 vs Flow 8/15；Pick & Place 只抓取 10/15 vs 9/15；完整任务 7/15 vs 1/15。
- 训练效率（LIBERO-Spatial）：ESP 15k 步 / 4.78 h 达 93.0%，Flow 60k 步 / 4.98 h 达 93.5%；单步 flow 比 ESP 快约 3.81×。

## 复现与解释边界

- 无公开代码；项目页的 Real-world Results 小节在抓取时无内容。
- 每步训练开销含 K 个候选与 O(K²) 两两项，更高；作者称样本效率抵消大部分，但降低训练成本留作未来工作。
- TwoBranch 为单 seed 示意；端到端未缓存时受视觉语言骨干限制，加速只有 1.40×。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-esp-energy-score-policy.md
- 概念入口：wiki/concepts/flow-matching-embodied-policy.md
- 方法入口：wiki/methods/diffusion-policy.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.07696>
- arXiv HTML：<https://arxiv.org/html/2610.07696v1>
- 项目页：<https://lilika-makabe.github.io/energy-score-policy-site/>

# WAM-Cache 论文来源归档

## 元数据

- **题目：** WAM-Cache: Staleness-Bounded KV Reuse for Efficient World Action Models
- **作者：** Kai Ding, Yang He, Ruijie Quan, Yi Yang（arXiv 页面未列单位）
- **预印本：** https://arxiv.org/abs/2610.11401 （v1，2026-10-08，cs.RO，交叉 cs.CV、cs.LG）
- **HTML：** https://arxiv.org/html/2610.11401v1
- **项目页：** https://dingkai0302.github.io/wam-cache/ （见 [wam-cache-project.md](../sites/wam-cache-project.md)）
- **代码：** https://github.com/DingKai0302/wam-cache （截至 2026-10-12 README 写 “Code coming soon”，未发布）

## 论文要解决的问题

WAM 用预训练视频 DiT 产生表征来引导动作专家。闭环控制中每个动作块都要跑一次视频 DiT 的 prefill，这是主要算力开销，而已有免训练加速方法在这一步仍是稠密计算。直接复用上一块的缓存又会把过时状态带入动作预测。

## 方法与训练信息

基座是 Fast-WAM：Wan2.2-5B 视频 DiT + 1B 动作专家 DiT，以 Mixture-of-Transformer 共享注意力；T5 文本编码器与视频 VAE 不变；多路相机拼成一帧后做 VAE 编码（RoboTwin 2.0 为 S_v=120 个视觉 token，LIBERO 为 98 个）；测试时只编码当前观测，每块一次 prefill，动作专家在其 K/V 上去噪。WAM-Cache 免训练，不改权重。

1. **每 token 状态：** 各层缓存的 K/V、上次刷新时的参考嵌入、年龄计数 g_i、动作专家上一块对该 token 的注意力质量 p_t[i]。
2. **刷新集 R_t = S_t ∪ P_t ∪ A_t：**
   - 潜变量惊异 S_t：当前嵌入与参考嵌入的 L2 距离（按两者范数均值归一化），取 top-b_s。
   - 动作注意力 P_t：动作专家各层、各动作 query 平均后对该 token 的注意力质量（取上一块第 1 个去噪步已算出的值，无额外开销），取 top-b_a。
   - 年龄界 A_t：g_i ≥ G 的 token 无条件刷新（G=0 关闭）。
3. **稀疏 prefill：** 首块稠密计算；之后只对 R_t 重算隐藏状态并原位覆盖 K/V，query 只为刷新 token 生成，仍对全部 S_v 个 K/V 做注意力；未选 token 沿用缓存。动作专家照常运行。成本按 ρ_t=|R_t|/S_v 线性缩放，ρ̄ ≤ b_s + b_a + 1/(G+1)。
4. **超参：** 去噪步数 N=2（也测 N=10），G=2 全程固定；(b_a,b_s) 默认 RoboTwin clean (0.4,0.1)、randomized (0.4,0.2)、LIBERO (0.5,0.1)、真机 (0.4,0.1)；较大预算为 (0.6,0.1)/(0.4,0.3)/(0.6,0.1)。

## 论文报告的结果

设置：bfloat16、单张 RTX PRO 6000 Blackwell；基线 FastV、ToMe、Eventful Transformer、VLA-Cache，均校准到相近 prefill FLOPs。

- **RoboTwin 2.0（50 任务，每任务 100 episode）：** 稠密 1.053 TFLOPs、clean 90.16% / randomized 89.66%。WAM-Cache clean 0.615 T（−41.6%）88.36%（−1.8），randomized 0.639 T（−39.3%）88.32%（−1.3）；对比同 FLOPs 下 FastV 62.72/60.18、ToMe 77.92/74.14、Eventful 52.94/51.36、VLA-Cache 86.16/84.26。较大预算：clean 90.08%（−28.2% FLOPs）、randomized 89.56%（−37.0%）。
- **LIBERO（四套件，每任务 50 episode）：** 稠密 0.859 T、平均 96.50%；WAM-Cache 0.582 T（−32.3%）95.85%；VLA-Cache 94.85%、Eventful 93.65%、ToMe 91.95%、FastV 78.80%；较大预算 96.50%（−26.1%）。Long 套件差距最大（90.40 对 93.80）。
- **真机（AIRBOT Play，Carrot in Bowl / Stack Cubes，各 20 次，每任务 100 条遥操作示范、训练 30k 步）：** 稠密 90.0/80.0（均值 85.0%，1.123 T）；WAM-Cache 90.0/75.0（82.5%，0.672 T，−40.2%）。
- **延迟（CUDA graph 回放）：** clean RoboTwin prefill 加速 1.23×（原生 120 token）、1.47×（480）、1.67×（1080，达到 1.71× FLOPs 上界的 98%）；randomized 1.58×。大分辨率结果为按原生刷新比例回放的成本测量，未训练对应 checkpoint。
- **刷新规则消融（clean RoboTwin，内容预算 0.5、G=2）：** Random 26.88%、像素差 60.40%、Oracle K/V 漂移 65.34%、仅惊异 60.62%、仅动作注意力 81.46%、注意力∪惊异 88.36%（ρ̄=0.584）。去掉年龄界后为 80.7%，G=2 增加 7.7 点；G=1→2 节省 10.2 点 FLOPs、仅多 0.6 点 SR。
- **去噪步数：** 稠密下 RoboTwin N=1 掉 20 点以上，N=2 比 N=10 低 1.3–1.5 点；N=10 时 WAM-Cache 距稠密 0.6（clean）/1.1（randomized）点内。

## 复现与解释边界

- 仅在 Fast-WAM 上验证；小视觉 token 数下实测延迟加速低于 FLOPs 降幅（固定开销）。
- 年龄界限制陈旧时长，不限制误差大小；Long 套件的缺口被归因于继承的注意力在阶段切换时滞后。
- 真机只有两任务、各 20 次，2.5 点差距落在抽样噪声量级（推测）。
- 代码未发布，无法核对实现；论文正文未给代码声明。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-wam-cache.md
- 基座：wiki/entities/paper-fast-wam.md
- 概念入口：wiki/concepts/world-action-models.md

## 一手来源

- <https://arxiv.org/abs/2610.11401>
- <https://arxiv.org/html/2610.11401v1>
- <https://dingkai0302.github.io/wam-cache/>

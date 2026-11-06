# World Models（Ha & Schmidhuber）论文通读

> 来源归档（原文通读 · 支撑 `wiki/overview/sun-awesome-wm-technology-map.md`）

- **标题：** World Models
- **作者：** David Ha、Jürgen Schmidhuber
- **类型：** paper / world-model / model-based-rl
- **原文：** <https://arxiv.org/abs/1803.10122>（v4，2018-05-09；2026-11 通过 arXiv PDF 获取全文）
- **一句话说明：** 把「大世界模型 + 极小控制器」拆成 V（VAE 压缩）/ M（MDN-RNN 预测）/ C（线性控制器）三件套，并展示智能体可完全在自己的「梦境」里训练再迁回真实环境；Awesome World Models 清单把它列为奠基工作。
- **核对范围：** 通读正文 §1 引言、§2 智能体模型（V/M/C）、§3 CarRacing 实验（含表 1）、§4 VizDoom 实验（含表 2、对世界模型的「作弊」与温度 τ 讨论）、§5 迭代训练、§6 相关工作与讨论，以及附录 A.1–A.3 的网络结构与训练设置开头；附录后半（CMA-ES 超参细节）与引用文献仅浏览，未逐项核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **三件套（§2）：** V 用 VAE 把每帧压成小隐向量 z（CarRacing 为 32 维、Doom 为 64 维）；M 是 MDN-RNN，建模 P(z_{t+1} | a_t, z_t, h_t)，输出高斯混合并用温度 τ 调节不确定性；C 是单层线性层 a_t = W_c[z_t h_t] + b_c。设计动机：把复杂度放在可用反向传播训练的 V、M 里，C 足够小，才能用 CMA-ES 这类进化策略解决信用分配难题。
- **参数量（§3.2、§4.2）：** CarRacing 中 VAE 4,348,547、MDN-RNN 422,368、控制器仅 867；Doom 中分别为 4,446,915、1,678,785、1,088。
- **训练流程（§3.2）：** 随机策略收集 10,000 条 rollout → 训 V → 用 V 编码数据训 M → 用 CMA-ES 优化 C。V、M 训练各不到一小时（单 GPU）；世界模型本身看不到奖励，只有 C 用奖励。
- **CarRacing 结果（§3.3、表 1）：** 仅 V：632 ± 251；V 加隐藏层：788 ± 141；完整世界模型（z 与 h 拼接）：906 ± 21，作者称是首个达到该任务「解决」线（100 次均值 900）的方法；对比 DQN 343、A3C 591–652、榜单最佳 838。
- **梦中训练（§4）：** M 额外预测 done 标志后即可包成 Gym 环境，C 完全在潜在空间的「幻觉」环境里训练，再迁回真实 VizDoom Take Cover，真实环境得分约 1100 步，超过「解决」线 750。
- **模型被利用（§4.5）：** 世界模型只是近似模型，控制器会找到欺骗它的对抗策略（如让梦里的怪物永不发射火球）；原因是策略会走到训练分布之外、模型出错的状态。缓解办法：用 MDN-RNN 的随机输出并提高温度 τ。
- **温度权衡（表 2）：** τ 过低（0.1、0.5）梦中分数约 2060–2086 但真实仅约 193–196（低于随机策略 210）；τ = 1.15 真实得分最高 1092 ± 556；τ = 1.30 降到 753 ± 139 但方差较小。τ 是需调的超参，在真实感与可利用性间折中。
- **迭代训练（§5）：** 复杂任务需循环：真实环境 rollout → 训 M 并在 M 内训 C → 回到真实环境；作者提出用好奇心 / 内在动机（翻转 M 的损失符号）驱动探索，仅作为未来方向，未实验。
- **相关工作（§6）：** 追溯到 Schmidhuber 1990–1991 的 RNN 世界模型与控制器、PILCO（高斯过程动力学）、基于自编码器压缩表征的控制等。

## 对 wiki 的映射

- [`wiki/entities/paper-ha-schmidhuber-world-models.md`](../../wiki/entities/paper-ha-schmidhuber-world-models.md)
- [`wiki/overview/sun-awesome-wm-technology-map.md`](../../wiki/overview/sun-awesome-wm-technology-map.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1803.10122>

# Flow-GRPO 原文通读（arXiv:2505.05470v5）

> 原始论文归档（paper；支撑 `wiki/entities/paper-flow-grpo.md`）。

- **英文标题：** Flow-GRPO: Training Flow Matching Models via Online RL
- **中文标题：** Flow-GRPO：通过在线强化学习训练流匹配模型
- **作者：** Jie Liu、Gongye Liu、Jiajun Liang、Yangguang Li、Jiaheng Liu、Xintao Wang、Pengfei Wan、Di Zhang、Wanli Ouyang
- **机构：** 香港中文大学 MMLab、清华大学、快手可灵团队、南京大学、上海人工智能实验室
- **会议：** NeurIPS 2025（主会）
- **论文：** https://arxiv.org/abs/2505.05470 （2025-05-08 v1；本次核对 2025-10-27 v5）
- **全文 HTML：** https://arxiv.org/html/2505.05470v5
- **论文 PDF：** https://arxiv.org/pdf/2505.05470
- **官方项目页：** https://gongyeliu.github.io/Flow-GRPO/
- **官方代码：** https://github.com/yifan123/flow_grpo （MIT；有训练入口）
- **原始资料类别：** paper / flow-matching / online-rl / text-to-image
- **收录日期：** 2026-11-10
- **核对范围：** 已通读 arXiv v5 摘要、`1–`6（动机、相关工作、流匹配 MDP 形式化、GRPO 目标、ODE→SDE、Denoising Reduction、三任务结果、消融、局限）、Table 1–4、Figure 1–7 图注、Appendix A（式 10–29 的 Fokker–Planck / score 推导）、Appendix B（模型/资源/超参）、Appendix C.1–C.7（对照实验、噪声/步数/组大小、FLUX、KL、定性图）、Appendix D（低步数样本对比）；参考文献核对相关方法来源，未逐篇通读全部引用论文。同步核查项目主页、官方 README 及 SD3 训练核心函数。
- **通读：** 已完成

## 研究问题与解决策略（`1–`4）

1. **目标场景：** 预训练文本生图 flow matching 采用确定性 ODE 去噪；online GRPO 需要多样化轨迹与可计算的条件采样对数概率。不能把「同一 prompt 改初始种子」误作每个中间 transition 都有非退化密度。
2. **流匹配基础：** 以数据 `x_0`、噪声 `x_1` 和 `t∈[0,1]` 定义 `x_t=(1-t)x_0+t x_1`，学习速度场 `v_θ(x_t,t)`，训练目标是回归 `x_1-x_0`。
3. **转 MDP：** 把 prompt、时刻和 latent 作为状态，下一步去噪 latent 为动作，终止时对图像打分。相同 prompt 采 `G` 条轨迹、按组内 reward 均值与标准差标准化优势；训练目标为 PPO-style importance ratio clip + 对参考流模型的 KL 约束。
4. **ODE → SDE（`4、Appendix A）：** 用 Fokker–Planck 方程对齐单时刻边缘分布，得到带可控 `σ_t` 的随机漂移。Rectified Flow 情况下，`dx_t=[v_t+(σ_t²/(2t))(x_t+(1-t)v_t)]dt+σ_t\,dW`；`σ_t=a√(t/(1-t))`。Euler–Maruyama 离散后为高斯 transition，可算对数似然、importance ratio 与同方差 KL。**边缘分布等价是理论构造，有限步长离散/拟合误差不能宣称严格相等。**
5. **Denoising Reduction：** **训练 rollout 10 步、评估/推理 40 步**。收集的 10 步图可能有伪影，但分组奖励仍足以提供训练信号；40→10 步数据采样在作者实验中缩短整体优化耗时，性能收敛无明显损失；不等于把线上推理改为 10 步。

## 论文实验事实（`5、附录）

| 场景 | 原始 SD3.5-M | Flow-GRPO（带 KL，若适用） | 口径 |
|---|---:|---:|---|
| GenEval overall | 0.63 | **0.95** | 组合图像生成正确率 |
| Visual Text Rendering | 0.59 | **0.92** | OCR 字符编辑距离所定义的任务准确率 |
| DrawBench PickScore | 22.34 | **22.37**（GenEval 优化） | 仅质量/偏好保护参照，不是任务准确率 |
| DrawBench DeQA | 5.39 | **5.25**（GenEval 优化） | 略低于基座，不是「所有质量指标均提升」 |
| PickScore 训练任务奖励 | 21.72 | **23.31**（带 KL） | 表 2 的任务指标 |
| FLUX.1 Dev PickScore | 22.62 | **23.97** | 附录 C.4，额外 backbone |

- **训练设置：** 主实验 SD3.5 Medium；`G=24`，`a=0.7`，512 像素，LoRA rank 32 / alpha 64；KL 系数 GenEval/OCR 为 0.04、PickScore 为 0.01。论文使用 **24 张 NVIDIA A800**，不是宣称单卡复现主论文训练。
- **效率：** 消融中数据收集 40→10 步，作者报告 **超过 4 倍的 reward-vs-wall-clock 收敛加速**；进一步降到 5 步不稳定，不能笼统写「越少越快」。
- **稳定性：** 组大小 6/12 的 PickScore 训练最终崩溃，24 保持稳定；过少噪声降低探索、过多噪声使奖励归零；合理 KL 减少画质退化/多样性坍缩，但会拖慢早期优化。
- **泛化：** 在未见物体类别的 GenEval overall 为 0.64→0.90；从 2–4 物体训练向 5–6 物体泛化，counting 为 0.13→0.48；属于图像任务划分。
- **基线：** 作者与 SFT、Flow-DPO、Flow-RWR（在线/离线）、改造的 DDPO、ReFL、ORW 比较；不能将不同奖励源/批次预算的数值不经校准混为绝对胜负。

## 关键风险和适用边界（`5.3、`6）

- **实证只覆盖文本到图像**：不是机器人动作策略、真机或视频生成实测。向 VLA / 人体动作流模型借鉴只是一条待验证的迁移方向。
- **奖励与样本分布依赖**：OCR 使用编辑距离与 OCR 打分，GenEval 使用检测器 / 属性规则，PickScore 使用偏好模型；奖励漏洞会导致偏移。
- **没有 KL 会失真**：GenEval、OCR 的 DrawBench 质量和偏好分下降；PickScore 任务会发生风格与跨种子多样性坍缩。论文摘要的「很少 reward hacking」需附此条件。
- **计算成本**：论文 24×A800 主实验；视频上扩展仍面临奖励构造、多个目标冲突、昂贵采样与 KL 权衡。

## 对 wiki 的映射

- [Flow-GRPO 主实体](../../wiki/entities/paper-flow-grpo.md)
- [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
- [强化学习](../../wiki/methods/reinforcement-learning.md)

## 原始出处

- NeurIPS 2025： https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a10c46572628d58cb44fb705f25cbbf-Abstract-Conference.html
- arXiv v5： https://arxiv.org/html/2505.05470v5
- 作者 GitHub： https://github.com/yifan123/flow_grpo
- 项目页： https://gongyeliu.github.io/Flow-GRPO/

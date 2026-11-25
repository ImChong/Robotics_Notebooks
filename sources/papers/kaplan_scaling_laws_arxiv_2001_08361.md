# Scaling Laws for Neural Language Models｜神经语言模型的规模法则

> 一手论文归档：Kaplan et al. (2020)，arXiv:2001.08361v1

- **标题：** Scaling Laws for Neural Language Models（神经语言模型的规模法则）
- **作者：** Jared Kaplan、Sam McCandlish、Tom Henighan、Tom B. Brown、Benjamin Chess、Rewon Child、Scott Gray、Alec Radford、Jeffrey Wu、Dario Amodei
- **机构：** Johns Hopkins University、OpenAI（以论文署名为准）
- **发表：** 2020-01-23，arXiv v1
- **类型：** paper / scaling-laws / language-modeling
- **arXiv：** <https://arxiv.org/abs/2001.08361>
- **全文 HTML：** <https://arxiv.org/html/2001.08361v1>
- **PDF：** <https://arxiv.org/pdf/2001.08361>
- **OpenAI 官方介绍：** <https://openai.com/index/scaling-laws-for-neural-language-models/>
- **入库日期：** 2026-09-21；**本次复核：** 2026-11-21
- **核对范围：** 通读 arXiv v1 HTML 正文 §1–§8、附录 A–D（含拟合表、compute-efficient frontier、局限与补充实验），并核对 OpenAI 2020-01-23 官方 Publication 页面；该官方页面仅有论文入口、摘要与作者，未额外披露参数表或复现代码。
- **通读：** 已完成

## 研究对象与实验口径

- **对象：** 自回归语言模型，主要为 decoder-only Transformer；另有 LSTM、Universal Transformer 对比。
- **数据：** WebText2，约 20.3M 文档、22.9B BPE tokens；还在 Books、Common Crawl、Wikipedia、Internet Books 等文本分布测迁移。
- **评测：** 主要是 1024-token 上下文的 **test cross-entropy（nats/token）**；非问答准确率或机器人成功率。
- **模型范围：** 非 embedding 参数量约 768–1.5B，数据子集约 22M–23B tokens（§3）。
- **变量：** N 为非 embedding 参数量；D 为独立训练数据 token 数；B 为每次更新 tokens；S 为参数更新步数；训练 FLOPs 近似 C≈6NBS。C_min 为批大小效率校正后的最低计算量（§5.1）。
- **训练条件：** 通常采用 Adam；超过 1B 参数的模型使用 Adafactor；多数实验 3000 步 warmup + cosine decay；§4 过拟合实验使用 10% dropout 和 test loss early stopping。

## 核心实证结果与公式

### 1）单轴幂律（§1.2、§3、附录 A）

其他轴不构成瓶颈时：

| 受限轴 | 原文拟合 | 指数 |
|---|---|---|
| 非 embedding 参数 N | L(N)≈(N_c/N)^α_N | α_N≈0.076 |
| 训练集 tokens D | L(D)≈(D_c/D)^α_D | α_D≈0.095 |
| 批大小校正后的计算 C_min | L(C_min)≈(C_c_min/C_min)^α_C_min | α_C_min≈0.050 |

指数对应本文数据、tokenization 与训练协议，**不是通用常数**。§4 联合拟合 Table 2 的 α_D=0.103 与 §1.2 的单轴拟合 0.095 口径不同，不能混用。

### 2）数据不足与过拟合（§4）

- 联合拟合：L(N,D)=[(N_c/N)^(α_N/α_D)+D_c/D]^α_D；N 与 D 未共同增加时出现边际收益递减。
- 本文设置与误差阈值下，接近收敛而避免明显过拟合的数据规模约 **D≳5×10³ N^0.74**（§4.2，式 4.4）。
- 这描述 **过拟合约束**，不等于固定计算预算下的最优 N:D 配比；论文亦专门提示两者不同。

### 3）训练步数、batch 与算力效率（§5–§6、附录 B）

- 临界批大小 B_crit(L)≈B_*/L^(1/α_B)，α_B≈0.21；在 B_crit 附近折中训练时长与计算效率。
- 无限数据近似下，学习曲线 L(N,S_min)≈(N_c/N)^α_N+(S_c/S_min)^α_S，α_S≈0.76（§5.2）。
- 固定 **C_min** 的经验最优分配：**N_opt∝C_min^0.73**、**B_crit∝C_min^0.24**、**S_min∝C_min^0.03**；不重复使用训练样本时的单 epoch 近似为 **D∝C_min^0.27**（附录 A，Table 6）。这里的重点是相对较大模型和早停。
- 附录 B.3 比较了同目标 loss 的两种收敛偏差：基于拟合模型，compute-efficient 方案对应约 2.7× 参数、7.7× 更少更新步数和 65% 更少计算；**这是拟合推算，不是实际硬件保证**。

### 4）迁移与模型形状（§3、附录 D）

- 固定非 embedding N 后，层数、宽度和 attention heads 等在常见范围内影响小；极端形状、极小模型除外。
- 其他文本测试集的损失随模型增长大致平行下降，存在分布偏移；**不能直接解释成所有下游任务也满足相同幂律**。
- LSTM 在更长上下文较后位置的 token 上劣于 Transformer；Recurrent Transformer 的同参数与同计算量排序并不一致。

## 论文明确指出的局限（§6.3、§8、附录 C）

1. 幂律缺乏统一理论解释，对极小 D、极长上下文与远距离外推没有保证。
2. 在极大算力的外推中，“数据使用量增长缓慢”与“避免过拟合所需数据量”出现**矛盾**，论文主动预告定律将失效；极限规模估计高度不确定。
3. 没有系统调优正则化、数据增强或所有优化器超参数，临界 batch 的域外外推有风险。
4. 只验证语言建模 loss，未建立具身成功率、工业节拍或机器人策略的缩放规律。
5. FLOPs 估计省略与 context 长度相关部分，超长上下文可能不再有效。

## 与知识库的映射及复现状态

- [唯一论文详情](../../wiki/entities/paper-scaling-laws-neural-language-models.md)
- [Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md) — 只能借用拟合与实验设计方法，不能把原文数值移植到具身任务。
- **开放状态：** OpenAI 官方发表页未列出训练/推理源码及数据下载入口，本次未据此断言第三方复现不存在。无可核验官方可运行入口，源码运行时序图不适用。

## 参考来源

- [Kaplan et al., arXiv v1 HTML](https://arxiv.org/html/2001.08361v1)
- [OpenAI 官方发表页（2020-01-23）](https://openai.com/index/scaling-laws-for-neural-language-models/)

# WePM：统一多模态模型的层级冲突协调（arXiv:2611.27365）

> 来源归档（paper）

- **英文标题：** WePM: Hierarchical Conflict Reconciliation for Unified MLLMs
- **中文标题：** WePM：统一多模态大语言模型的层级冲突协调
- **副标题：** Unified Pocket Models for In-App Multimodal Intelligence
- **作者：** Zhiqiang Yuan、Qiwei Yan、Liwei Yang 等 12 位作者
- **机构：** 微信 AI / 腾讯（WeChat AI, Tencent）、中国科学院大学、西安交通大学
- **arXiv：** https://arxiv.org/abs/2611.27365
- **HTML：** https://arxiv.org/html/2611.27365v1
- **PDF：** https://arxiv.org/pdf/2611.27365
- **版本：** arXiv:2611.27365v1，2026-11-20；15 页、6 图、7 表，预印本
- **代码：** https://github.com/Tencent/WePocketModel
- **模型：** https://huggingface.co/collections/tencent/wepocketmodel
- **入库日期：** 2026-11-23
- **核对范围：** 已通读官方 arXiv HTML v1 的 Abstract、§1–7（问题形式化、MGVB/ULKD/SEMO、实验设置及单任务/联合任务比较、消融、HCD、相关工作、Limitations 和 Conclusion），核对 Figure 1–6 对应叙述、Tables 1–7 主要数据与文献列表；对照官方 GitHub README、推理入口及 HF 合集。未复测训练/推理结果。
- **通读：** 已完成
- **一句话说明：** 同一 Qwen3.5 视觉语言骨干共享检索向量、自动回归生成与结构化决策，通过多粒度视觉预算、双教师软监督及任务曝光/损失权重的顺序梯度累积缓解异质任务冲突。

## 核心摘录

1. **指标（§1，式 1）：** HCD = K / M × 任务族归一化表现的加权调和均值；K 按 representation / generation / decision 的能力族计数，M 是驻留推理权重内存，而非总 RAM/显存。OCR、VQA 和 caption 同属生成族，不能拆开刷 K。
   - **对 wiki 的映射：** [WePM 独立论文页](../../wiki/entities/paper-wepm-hierarchical-conflict-reconciliation.md) 定义 HCD 与适用口径。
2. **视觉颗粒度冲突（§2.2、§3.1）：** 一次视觉编码后的 token 被 MGVB 分成 global、regional、fine 三个 bank，预算满足 B_g+B_r+B_f=B；区域与 patch 选择可用问题文本做条件，只有图像的离线索引使用 query-independent fallback，避免未来 query 污染索引。
   - **对 wiki 的映射：** [多模态基础](../../wiki/concepts/multimodality-basics.md) 与 WePM 方法流程图。
3. **监督冲突（§3.2）：** ULKD 不要求把每条检索 query 人工补成唯一答案；冻结的生成教师通过学生采样前缀上的 token KL 提供 OPD，嵌入教师通过候选 listwise KL + 投影的 embedding L2 提供软监督；决策仍使用自身 typed loss。
   - **对 wiki 的映射：** WePM 页的「核心机制」及「与其他工作对比」。
4. **任务更新冲突（§3.3）：** SEMO 固定每任务损失系数 λ_i，同时通过每轮包含的任务 batch 数 n_i,t 控制 exposure；逐 batch 前反向、保留梯度，所有任务 batch 结束后才执行一次 optimizer step。它不做梯度冲突方向投影。
   - **对 wiki 的映射：** WePM 页的「核心机制」及顺序训练/推理边界。
5. **训练设置（§4.1）：** 以 Qwen3.5-2B/4B dense 初始化并全参数微调，约 1 亿清洗样本、最大 8192 token；完整任务族生成/嵌入/决策抽样比 7:5:1；生成教师 Qwen3.5-27B、嵌入教师 WeMM-9B。**7:5:1 是曝光比，不是 λ 损失权重。**
   - **对 wiki 的映射：** 实验条件与复现限制。
6. **单任务结果（§4.2，Tables 1–3）：** WePM-4B 的 30 个 embedding 平均 72.86；39 个 generation 平均 62.41；7 个 typed decision 平均 61.11。相邻强基线分别 72.82、61.83、61.06，差距很小；分数是不同 benchmark 量纲的组内均值，不能当机器人成功率。
   - **对 wiki 的映射：** 实验与评测表。
7. **统一方法对比（§4.3，Table 4）：** 仅在 representation+generation 共同能力域对比 CAFe/FROMAGe/MMGen/ULME，最强 ULME 的 Joint 38.15，WePM 52.74；这里没有把 decision 纳入 Joint。
   - **对 wiki 的映射：** 与其他工作对比中声明评测协议。
8. **消融（§4.4，Tables 5–7）：** MGVB B=128/256/512 对应 Joint 39.58/41.55/41.90，128 时 OCR 58.53 低于无 MGVB 基线 64.59；ULKD 的固定 pseudo-target 49.33 vs OPD 51.70；顺序累积 76.23 GiB 峰值 vs 同步图约 209.98 GiB，均为特定训练设置与显存计量。
   - **对 wiki 的映射：** MGVB 的保细节与 SEMO 显存取舍。
9. **驻留内存（§4.5，Fig.6）：** WePM 4B BF16 权重约 8 GB，独立三模型最省内存组合为 21 GB；HCD 0.244 vs 0.092 即 2.64×，权重驻留减少 61.9%。不含 KV cache、激活、框架和碎片。
   - **对 wiki 的映射：** 工程实践中分开评估 weight memory 与峰值 runtime。
10. **局限（§6）：** 相比原始 Qwen 骨干，agentic/tool-use 能力存在退化；静态 loss/exposure 需根据数据组合先验设定，未验证其适应任意任务动态变化；无机器人闭环运动、真实 VLA 训练与控制安全实验。
    - **对 wiki 的映射：** [模型架构纵深 Stage 5](../../roadmap/depth-model-architecture.md) 仅作为机载多模态层候选，不宣称低层控制可用。

## 开放状态与证据边界

- 官方 [WePocketModel 仓库归档](../repos/wepocketmodel.md)：已发布 **推理 runtime、Transformers/CPU GGUF 调用样例及脚本**，未在公开仓库中核对到对应论文的完整 MGVB/ULKD/SEMO **训练复现脚本**。
- 官方 [Hugging Face 模型合集归档](../models/wepocketmodel-collection.md)：2B/4B Transformers 与 2B/4B GGUF 可见；具体权重许可依各 checkpoint 卡片，不从代码 Apache-2.0 推断。
- 论文分数均为作者报告，**本次未独立运行权重、未在设备上测峰值显存/吞吐、未跑机器人闭环**。

## 对 wiki 的映射

- [WePM 论文详情](../../wiki/entities/paper-wepm-hierarchical-conflict-reconciliation.md)
- [模型架构路线 Stage 5](../../roadmap/depth-model-architecture.md)

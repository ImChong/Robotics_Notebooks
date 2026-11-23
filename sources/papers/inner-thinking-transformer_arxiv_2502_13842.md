# Inner Thinking Transformer：按 token 动态分配思考深度（arXiv:2502.13842）

> 来源归档（paper）

- **标题：** Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking
- **中文标题：** 内部思考 Transformer：通过动态深度缩放实现自适应隐式推理
- **类型：** paper
- **作者：** Yilong Chen, Junyuan Shang, Zhenyu Zhang, Yanxi Xie, Jiawei Sheng, Tingwen Liu, Shuohuan Wang, Yu Sun, Hua Wu, Haifeng Wang
- **机构：** 中国科学院信息工程研究所、中国科学院大学、百度、北京师范大学
- **会议：** ACL 2025 Long Paper，DOI 10.18653/v1/2025.acl-long.1369
- **arXiv：** https://arxiv.org/abs/2502.13842
- **PDF：** https://arxiv.org/pdf/2502.13842 （v2，2025-02-23）
- **正式发表：** https://aclanthology.org/2025.acl-long.1369/
- **代码：** 论文 PDF 与 ACL Anthology 记录未给 ITT 官方实现仓库链接；§4.1 使用第三方 Sheared-LLaMA / Composer 实验栈，不等同于 ITT 源码已公开（核对日 2026-11-23）。
- **入库日期：** 2026-11-23
- **核对范围：** arXiv v2 PDF 共 15 页：摘要、§1–6、实验表 1–3、图 1–7、Limitations、附录 A.1–A.4（算法 1、补充论证、路由消融表 4）及文献目录；对照 ACL Anthology 发表信息。
- **通读：** 已完成
- **一句话说明：** ITT 对 Transformer 的部分层引入 token 重要性路由，仅对选中 token 重复共享层计算，并用思考步编码和残差累计融合，在不增模型参数的前提下以额外 FLOPs 换取更高语言模型准确率。

## 核心摘录

1. **动机（§2，图 2）：** GPT-2 在 AQuA 的困难样本上不同注意力层的梯度核范数波动与尖峰更明显；这是作者提出按 token 分配额外层内计算的实证动机，不是严格证明「高梯度就等于需要思考」。
   - **对 wiki 的映射：** [ITT 论文独立节点](../../wiki/entities/paper-inner-thinking-transformer.md) 解释机制和证据边界。
2. **思考步与 RTC（§3.1–3.2，式 2–4）：** 将层内每次变换视作一个潜在思考步骤；共享变换 f 重复执行，累积各步输出并按可学习的 Thinking Step Encoding ϕ^(t) 加权。相较只循环同一层，RTC 能组合多个增量贡献。
   - **对 wiki 的映射：** [ITT 论文独立节点](../../wiki/entities/paper-inner-thinking-transformer.md) 的流程图。
3. **Adaptive Token Routing（§3.3，式 5–7、图 3）：** 线性预测器给每个 token 的隐藏状态打分，Top-K/分位阈值选择重要 token；选中者增加一次带权共享层处理，未选中者沿用旧表示，再由 RTC 汇总。每一轮重新路由，推理阶段可调整选中比例和轮数。
   - **对 wiki 的映射：** [Transformer 基础](../../wiki/concepts/transformer.md) 的动态深度扩展。
4. **实验设置（§4.1）：** 基于 LLaMA2 风格构造 162M/230M/466M 参数模型；RedPajama 预训练 50B tokens，8×A100 80GB，序列长 4096；每隔一层替换为 Loop/ITT 层；比较 2×/3×/4× 深度。
   - **对 wiki 的映射：** [模型架构路线](../../roadmap/depth-model-architecture.md) Stage 5，按精度与计算量共同读论文。
5. **结果（§4.2，表 1–2）：** 162M ITT×4 平均准确率 42.1，466M dense 43.6，即约 96.6%（摘要四舍五入报告 96.5%）；前者 FLOPs 3.29、后者 4.92（论文表中单位）；同规模 162M dense 准确率 40.4、FLOPs 1.88。训练达到同规模 dense 目标表现只需 56.8% token 预算，节省 43.2%；该比例不能泛化为全训练成本节省。
   - **对 wiki 的映射：** [ITT 论文独立节点](../../wiki/entities/paper-inner-thinking-transformer.md) 的实验与成本说明。
6. **消融（§4.3，表 3）：** ITT×4-162M PPL 10.25；移除 RTC→11.02（+0.77），移除 ATR→10.44（+0.19），移除 Step Encoding→10.56（+0.31；正文有一处 +0.31，表格列 +0.22，与 10.25 算术比较亦为 +0.31，应以表中原始值为准并记录不一致）。训练中固定路由容量设定，推理可弹性变化。
   - **对 wiki 的映射：** [ITT 论文独立节点](../../wiki/entities/paper-inner-thinking-transformer.md) 记录消融与原文口径。
7. **局限（Limitations）：** 训练期使用固定路由模式/预算设置，最高只验证 466M，RTC 反向传播增加内存开销；实验是语言模型测评，没有机器人控制、真实机载推理时延或闭环成功率验证。
   - **对 wiki 的映射：** [模型架构路线](../../roadmap/depth-model-architecture.md) 将它作为迁移前需验证的架构候选，而非已验证控制策略。

## 开源与复现边界

- **未找到官方 ITT 代码入口：** arXiv v2 正文、附录、ACL 官方条目均未列可直接获取的 ITT 实现仓库。Sheared-LLaMA 仅是作者声明采用的实验基础设施，不应标成 ITT 官方代码。
- **源码运行时序图：** 不适用（缺少可核对的 ITT 官方可运行源码与入口）；仅绘制基于论文 §3 / 附录算法 1 的方法流程图，不伪造运行调用链。
- **解释边界：** ITT 的「inner thinking」指隐表示的多次层内变换，不等同显式 Chain-of-Thought 文字推理，也不意味着有机器人环境模型。

## 对 wiki 的映射

- [独立详情节点：Inner Thinking Transformer](../../wiki/entities/paper-inner-thinking-transformer.md)
- [模型架构纵深路线 Stage 5](../../roadmap/depth-model-architecture.md)

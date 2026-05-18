# CapVector: Learning Transferable Capability Vectors in Parametric Space for Vision-Language-Action Models（arXiv:2605.10903）

> 来源归档（ingest）

- **标题：** CapVector: Learning Transferable Capability Vectors in Parametric Space for Vision-Language-Action Models
- **缩写：** **CapVector**
- **类型：** paper / vision-language-action / finetuning / model-merging
- **arXiv：** <https://arxiv.org/abs/2605.10903>（PDF：<https://arxiv.org/pdf/2605.10903>）
- **项目页：** <https://capvector.github.io/>
- **代码：** <https://github.com/OpenHelix-Team/CapVector>（论文元数据标注）
- **权重集合：** <https://huggingface.co/haofuly/capvector_models_collection>（论文元数据标注 *ready to use*）
- **作者：** Wenxuan Song, Han Zhao, Fuhao Li, Ziyang Zhou, Xi Wang, Jing Lyu, Pengxiang Ding, Yan Wang, Donglin Wang, Haoang Li（HKUST(GZ)、浙江大学、西湖大学、清华大学、北京智源人工智能研究院等；贡献与通讯作者以 PDF 为准）
- **入库日期：** 2026-05-18
- **一句话说明：** 在 **参数空间** 把「**辅助目标 SFT** 带来的通用能力提升」与「**标准 SFT** 对任务动作分布的拟合」**解耦**：在同一小规模能力抽取集上分别训出 \(\theta_{\text{ao}}\) 与 \(\theta_{\text{ft}}\)，用 **\(\gamma=\theta_{\text{ao}}-\theta_{\text{ft}}\)** 解释为 **capability vector**，合并得 **\(\theta_{\text{meta}}=\theta_{\text{pt}}+\alpha\gamma\)**；下游仅用 **标准 SFT + 轻量正交正则** 即可在开销上接近纯 SFT 的同时保留辅助训练的收益，并在多 VLA 骨干与跨仿真/真机设置上报告泛化。

## 摘录 1：问题与动机（摘要 / 引言）

- **痛点：** 预训练 VLA 在复杂下游上 **纯演示 + 标准 SFT** 往往 **收敛慢、增益有限**；引入 **空间感知、长程推理** 等 **辅助训练目标** 的 SFT 虽能 **减步数、抬性能**，但常带来 **额外模块、额外前向与对齐计算**，任务与数据规模上来后 **开销难承受**。
- **核心问法：** 能否把精心设计微调流程带来的有益性质 \(s\) **写回预训练骨干自身**，使后续 **只依赖标准 SFT** 就继承 **同等训练效率与性能**，而 **不持续支付** 辅助目标的计算成本？

**对 wiki 的映射：** 与 [VLA](../../wiki/methods/vla.md) 中「数据规模 ↑、推理延迟、后训练栈复杂度」并列，把 **后训练目标设计** 从「在线辅助损失」推进到 **离线参数算术 + 下游轻正则** 的另一条轴。

## 摘录 2：能力向量定义与两阶段机制（§2 要点）

- **记号：** 预训练 \(\theta_{\text{pt}}\)；在能力抽取集 \(\mathcal{D}_{\text{ext}}\) 上，**标准 SFT** 得 \(\theta_{\text{ft}}=\theta_{\text{pt}}+\Delta_{\text{ft}}\)；**带辅助目标的 SFT** 得 \(\theta_{\text{ao}}=\theta_{\text{pt}}+\Delta_{\text{ao}}=\theta_{\text{pt}}+\delta_{\text{ao}}+\gamma_{\text{ao}}\)（论文将 \(\delta\) 解释为任务相关更新、\(\gamma\) 为辅助目标诱导的 **capability** 分量）。
- **关键假设（经验支撑）：** 在 **一致微调设置** 下 \(\Delta_{\text{ft}}\approx \delta_{\text{ao}}\)，从而 **\(\gamma_{\text{ao}}=\theta_{\text{ao}}-\theta_{\text{ft}}\)**，即 **两策略 checkpoint 的逐参数差** 即能力向量。
- **合并：** \(\theta_{\text{meta}}=\theta_{\text{pt}}+\alpha\gamma_{\text{ao}}\)，作为任意新任务 **标准 SFT** 的更好初始化。
- **防遗忘：** 下游更新 \(\Delta'_{\text{ft}}\) 与 \(\gamma\) **共空间** 时易 **冲掉** 已注入能力；论文引入 **逐元素乘积和形式的正交正则** \(\mathcal{L}_{\text{orth}}\)，总损失 \(\mathcal{L}=\mathcal{L}_{\text{action}}+\lambda\mathcal{L}_{\text{orth}}\)；**LoRA** 场景论文说明主要在 **矩阵 A** 上计算该正则（与 O-LoRA 类讨论一致）。

**对 wiki 的映射：** 沉淀方法级叙事到 [`wiki/entities/paper-capvector-capability-vectors-vla.md`](../../wiki/entities/paper-capvector-capability-vectors-vla.md)；与 **task vector / model merging** 文献脉络在实体页「推荐继续阅读」中给出入口。

## 摘录 3：实验设置与主要结论（§3 摘要）

- **仿真：** **LIBERO** 四套件与 **RoboTwin 2.0**（论文报告 10 个净背景任务为主评测，另用 5 任务等作能力抽取）；**骨干** 覆盖 **OpenVLA-OFT**、**StarVLA**、**\(\pi_{0.5}\)**；**辅助 SFT** 以 **Spatial Forcing**、**LaRA-VLA** 为代表。
- **ID：** 与全量 **Spatial Forcing** 相比，自 **\(\theta_{\text{meta}}\)** 出发的 **CapVector + 标准 SFT** 在多种训练步数下 **成功率接近或更优**；**去掉正交损失** 在 **长训练（如 150k steps）** 上出现 **能力退化**，加回后与带辅助目标基线对齐或超过。
- **OOD / 多样性：** 用 **LIBERO 上抽取的 \(\gamma\)** 在 **RoboTwin** 等 **跨域下游** 仍相对 **纯标准 SFT** 明显提升（论文 Table 2 叙事）；在 **StarVLA + LaRA-VLA**、**\(\pi_{0.5}\)+Spatial Forcing** 等组合上展示 **不同辅助目标** 均可抽取并合并。
- **真机：** 论文声称 **内外部真机** 实验支持 **开箱跨环境/本体** 的实用性（细节以 PDF § 与附录为准）。

**对 wiki 的映射：** 实体页表格化 **骨干 × 辅助方法 × 抽取域 / 下游域** 读论文时的定位锚点；与 [RoboTwin](../../wiki/entities/robotwin.md)、[StarVLA](../../wiki/methods/star-vla.md) 等现成节点互链。

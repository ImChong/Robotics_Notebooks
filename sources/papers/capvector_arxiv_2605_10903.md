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
- **本次复核：** 2026-11-15；一手来源为 [arXiv HTML v1](https://arxiv.org/html/2605.10903)（2026-05-11）与 [CapVector 官方项目页](https://capvector.github.io/)；未将摘要代替全文。
- **核对范围：** 通读 arXiv HTML v1 的 §1–6（背景、CapVector 两阶段公式、RQ1–RQ5 实验、相关工作、结论、致谢）及附录 §7–12（合并权重、正交损失及额外开销、真机设置、局限与基线）；核验作者项目页与官方 GitHub README、`capvector-pi05/capvector/compute_param_diff.py`、`apply_param_diff.py`、`scripts/train_regular_loss_pytorch.py`、`scripts/serve_policy.py`、`examples/simple_client/main.py` 的关键入口。正文与附录通过 HTML 原文核对，未声称已打开 PDF，也未运行模型训练/真机评测。
- **通读：** 已完成
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


## 2026-11-15 原文完整复核补充（基于 arXiv HTML v1）

- **正确的正交正则（§2.3 Eq.7）：** 作者不是对整组矩阵只做点积平方，而是逐元素绝对乘积求和：`L_orth = Σ_p Σ_ij |γ_ao^(p)[i,j] · Δ'_ft^(p)[i,j]|`。LoRA 仅计算 A 矩阵相关部分，避免与笼统“正交”说法混淆。
- **论文 Table 1 / LIBERO（§3.2）：** 5k steps 的四套件平均成功率，Spatial Forcing 87.5%、OpenVLA-OFT 82.1%、CapVector（带正则）91.7%；150k steps 依次为 96.9%、92.7%、97.1%。均为作者结果，不能与其他任务协议混排。
- **论文 Table 2 / RoboTwin 10 任务（§3.3–3.4）：** OpenVLA-OFT 6.7%，Spatial Forcing 33.1%，CapVector 从 LIBERO-Spatial 抽取为 31.8%；`π_0.5` 基线 15.1%，SF 23.6%，CapVector VLM+Expert 23.5%。这组结果说明并非所有指标都超过辅助目标基线。
- **论文 Table 3 / StarVLA（§3.4）：** LaRA-VLA 97.9%、StarVLA 94.5%、CapVector 97.1%（LIBERO 四套件平均）；不同 VLA 架构上有复用能力，但仍需检查各自实验设置。
- **能力抽取数据质量（§3.5，Figure 3、Table 4）：** 在 RoboTwin 控制任务数/样本量后，更丰富的背景与物体组合更有利于能力向量泛化；只增加任务间外观差异可能引入 shortcut，故“数据越多越好”不是本文结论。
- **超参与开销（附录 §7–9）：** 基于 OpenVLA-OFT LoRA 的合并系数最优示例 α=1.1（仅此消融配置），正文默认正则权重 λ=1e-4。附录报告普通 OFT 基线 17.9T FLOPs / 62.8G 显存，Spatial Forcing 额外 5.0T FLOPs / 10.9G，CapVector 正则额外 0.3G FLOPs / 0.5G；这些是论文所测训练设置，不是抽取两份 checkpoint 的免费成本。
- **真机与局限（§3.6，附录 §10–11）：** 作者对 UR3、ARX Lift 2、AgileX Cobot 的演示和跨本体验证，外部试管搬运任务成功率 0.36→0.53；适用范围聚焦**基于 SFT 的能力向量**，RL 场景留待未来研究。
- **工程核查：** 官方仓库 `capvector-pi05/` 已公开差分提取、合并、正交正则训练与 WebSocket 伺服入口；`examples/simple_client/main.py` 采用**随机合成观测**作接口与延迟示例，不可称该脚本已真实控制机器人。

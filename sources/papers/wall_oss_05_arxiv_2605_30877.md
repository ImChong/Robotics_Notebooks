# Wall-OSS-0.5 Technical Report（arXiv:2605.30877）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2605.30877>（v1 提交 2026-05-29，v2 2026-06-01）
- **HTML：** <https://arxiv.org/html/2605.30877v2>
- **PDF：** <https://arxiv.org/pdf/2605.30877>
- **作者：** Ryan Yu, Pushi Zhang, Starrick Liu, Brae Liu, Miracle Kang 等（X Square Robot Team，共 27 位作者）
- **代码：** <https://github.com/X-Square-Robot/wall-x>；权重 <https://huggingface.co/x-square-robot/wall-oss-0.5>
- **入库日期：** 2026-11-05
- **再核日期：** 2026-11-21
- **线索来源：** LeRobot 固定提交 `9e5ee98` 的 `docs/source/wall_oss_05.mdx`（`policy.type=wall_oss_05`，检查点 `lerobot/wall-oss-0.5`）
- **核对范围：** 通读 `https://arxiv.org/html/2605.30877v2` 全文，含摘要、§1 引言、§2 方法（MoT、Vision-Aligned RVQ、梯度桥、Action-Space Supervision、26 维动作接口、DMuon、CUDA Graph 15 Hz 推理优化）、§3 数据配方（>1M traj/epoch, 90M 多模态含 12M bridge）、§4 实验（17 任务零样本、15 任务微调、5/10/19 任务扩展、5 项多模态评测）、§5 消融、§6 相关工作、§7 讨论与局限；未运行真机实验
- **通读：** 已完成

## 核心摘录（通读 arXiv:2605.30877v2 全文）

- **架构与模型规模：** Qwen2.5-VL-3B-Instruct 初始化，每层采用 Mixture-of-Transformers（MoT）拆分 VL Expert 与 Action Expert（共享注意力，总参数 >4B）；离散动作与连续动作通过注意力掩码互相屏蔽，但流匹配梯度不阻断回传。
- **梯度桥共训（§2.2.1）：** 单阶段联合优化 $\mathcal{L}=\mathcal{L}_{\text{flow}}+\lambda_{\text{act}}\mathcal{L}_{\text{act-CE}}+\lambda_{\text{mm}}\mathcal{L}_{\text{mm-CE}}$（$\lambda_{\text{act}}=\lambda_{\text{mm}}=0.01$，动作/多模态 batch 比 9:1）。Vision-Aligned RVQ 替代 FAST 作为 VLM 原生动作语义梯度桥；流匹配对 VLM 主干更新占比稳定在约 5%。
- **Action-Space Supervision（§2.2.2）：** 输出仍为速度场，但损失定义在恢复动作 $\hat A=A_\tau+(1-\tau)f_\theta(A_\tau,\tau)$ 上，等价于 $(1-\tau)^2$ 权重，聚焦高噪声段低频轨迹形状；LIBERO 消融峰值 96.5%（25k 步）高于速度空间。
- **系统与评测（§2.3, §4–§5）：** RTX 5090 三视角 448px 推理约 15 Hz（CUDA Graph + 算子融合，较 eager 快 4×）。预训练 checkpoint 17 任务零样本平均 51.1（400k 步，Seen 50.0 / Unseen 53.6，Rope Tightening 82）；15 任务微调平均 60.5% 对 $\pi_{0.5}$ 43.0% 和 DreamZero 33.4%。多模态方面 Embodied Grounding +21.8、Where2Place +11.0，但通用 RealWorldQA -15.0、ERQA -5.5。
- **局限（§7）：** 仅在 3B 主干验证；单帧输入缺时序记忆；绑定 26 维末端动作，难以直接覆盖灵巧手；依赖人工评分 rubric。

## 对 wiki 的映射

- [paper-wall-oss-0-5](../../wiki/entities/paper-wall-oss-0-5.md)
- 一代 WALL-OSS：[cn-os-wall-x](../../wiki/entities/cn-os-wall-x.md)
- LeRobot 实现：[lerobot.md](../repos/lerobot.md)

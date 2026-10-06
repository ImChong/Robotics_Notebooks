# StableMimic（arXiv:2608.02385）

> 来源归档（ingest）

- **标题：** StableMimic: Smooth Human-Like Recovery for Humanoid Motion Tracking
- **arXiv：** <https://arxiv.org/abs/2608.02385> · <https://arxiv.org/pdf/2608.02385>
- **作者：** Weihao Wu、Ming Huang、Ruofei Liu、Jinglei Nie、Shuxiang Guo、Chunying Li
- **入库日期：** 2026-10-05
- **摘要：** 跟踪/恢复专家与本体门控混合策略，用于动作跟踪和跌倒恢复。训练测试协议及限制见论文。
- **详情：** [StableMimic](../../wiki/entities/paper-stablemimic.md)

## 核心摘录

> 2026-10-05 核对 arXiv HTML 版。

- 评测协议：Unitree G1 + retargeted LAFAN1（dance 子集用于跟踪、get-up 子集只用于训练），统一在 MuJoCo 中评测；100 次 525–575 N、0.2 s 的四向躯干推力扰动。
- Table III：StableMimic (MoE) MPBPE 28.53 mm、MJAE 88.83×10⁻³ rad，优于 BeyondMimic、KungFuAthlete、BFM-Zero 和 Single-MLP 消融。
- Table IV：恢复成功 100/100（KungFuAthlete 100、Single-MLP 98、BFM-Zero 94、BeyondMimic 0），跌倒后肢体速度、行程、力矩和正功最低。
- 真机 G1 只做定性展示（dance、常值站立参考），作者声明不是安全认证。

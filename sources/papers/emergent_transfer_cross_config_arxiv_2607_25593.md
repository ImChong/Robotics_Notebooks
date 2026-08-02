# When Does Legacy Data Start to Help? Emergent Transfer in Cross-Configuration Robot Learning（arXiv:2607.25593）

> 来源归档（ingest）

- **标题：** When Does Legacy Data Start to Help? Emergent Transfer in Cross-Configuration Robot Learning
- **类型：** paper / cross-configuration transfer / VLA co-training / hardware iteration / emergent transfer
- **arXiv：** <https://arxiv.org/abs/2607.25593>（PDF：<https://arxiv.org/pdf/2607.25593.pdf>）
- **作者：** Tao Wang\*、Hudson Hou\*、Yingdong Hu、Yufeng Liu、Qinghai Li、Yingjie Jiang、Yingzhi Wang、Cheng Ma、Richard Wang（项目牵头）、Yang Gao（通讯）
- **机构：** 华中科技大学（HUST）；千寻智能（Spirit AI）；北京大学（PKU）；上海交通大学（SJTU）；哈尔滨工业大学（HIT）；清华大学（Tsinghua）
- **项目链接：** <https://arxiv.org/abs/2607.25593>（截至入库日**无独立项目页 / 无官方代码仓**）
- **入库日期：** 2026-08-02
- **一句话说明：** 在轮式人形两代硬件（相机 + 夹爪更换、形态不变）上，用 π₀.₅ VLA 细调实证：遗留示教并非始终有用，存在任务依赖的迁移阈值 τ(T) 与三相涌现迁移；并给出 phase-aware 新机数据采集规则。

## 开源状态（核查，2026-08-02）

- **确认未开源：** 用户给定「项目链接」即为 arXiv 摘要页；PDF / HTML 全文未列 GitHub、Hugging Face 或独立项目页；亦无 “code will be released” 承诺。
- **可复现边界：** 骨干为 Physical Intelligence **π₀.₅**（公开权重见 [openpi](https://github.com/Physical-Intelligence/openpi)），但本文 teleop 数据、双代机体配置与微调配方**未随文发布**。
- **源码运行时序图：** wiki 实体页标 **不适用**。

## 摘要级要点

- **问题：** 硬件迭代后，旧配置（legacy）示教何时开始帮助新配置？「多混旧数据一定更好」是否成立？
- **设定：** 同一轮式人形形态，Gen-1→Gen-2 更换相机（单目 RGB→鱼眼）与夹爪（位置控→力/位混合 + 腕力传感）；主实验固定底座桌面操纵（笔/花抓取与插入），留出移动双臂浇水任务做 held-out 验证。
- **方法：** 从预训练 π₀.₅ 做 BC 微调；单配置 vs 等概率采样的跨代 co-training；无显式硬件代际标签。
- **发现：** 共训练增益 ΔSR 相对新机 standalone 成功率呈倒 U：
  - **Phase I（&lt;~15–20%）：** 无收益（花插入 10%→10%）
  - **Phase II（~20–75%）：** 大幅收益（花插入 23.3%→86.7%；笔插入 71.7%→98.3%）
  - **Phase III（&gt;~75%）：** 收益衰减（笔插入 85%→93.3%）
- **理论：** 任务阶段可解码性驱动梯度对齐 → 迁移阈值 τ(T)；残差策略不确定性解释饱和区收益下降。
- **工程规则：** 先估任务复杂度 H(T)=L log(1/ε)，估 τ̂(T)；若 SR₂&lt;τ̂ 则继续采新机数据，否则再混 legacy。浇水任务上把新机采集从 8h 降到约 1.5h 即可进入高收益区。

## 核心论文摘录（MVP）

### 1) 三相涌现迁移（Emergent Transfer）

- **链接：** §1；§4；Table 2–4；Figure 2
- **摘录要点：** 遗留数据在低能力区无效，越过阈值后增益陡升，接近饱和后边际下降；作者称中间跃迁为 emergent transfer（类 grokking，但横轴是目标配置 standalone 性能而非模型规模/训练时长）。
- **对 wiki 的映射：**
  - [Emergent Transfer 实体页](../../wiki/entities/paper-emergent-transfer-cross-config.md)
  - [跨具身迁移 hub](../../wiki/overview/hub-cross-embodiment.md)

### 2) 迁移阈值 τ(T) 与梯度对齐解释

- **链接：** §5；Definition 1；Theorem 1–2；Appendix C
- **摘录要点：** τ(T) 定义为期望 ΔSR&gt;0 的最小 standalone SR；阶段结构未形成时 legacy/target 梯度对齐期望 ≤0；越过后同阶段监督对齐，且增益随 (1−SR) 下降形成倒 U。
- **对 wiki 的映射：**
  - [Emergent Transfer 实体页](../../wiki/entities/paper-emergent-transfer-cross-config.md)
  - [跨具身策略迁移选型指南](../../wiki/queries/cross-embodiment-transfer-strategy.md)

### 3) Phase-aware 数据采集规则（浇水 held-out）

- **链接：** §6；Table 5
- **摘录要点：** 0.5h 新机仍 Phase I（无增益）；1.5h 进入 Phase II（浇水子阶段约 +38–40pp）；8h 已近 Phase III（增益落入噪声）。规则目标是「采够让 legacy 变有用」，而非「尽可能多采新数据」。
- **对 wiki 的映射：**
  - [Emergent Transfer 实体页](../../wiki/entities/paper-emergent-transfer-cross-config.md)
  - [人形训练数据管线](../../wiki/queries/humanoid-training-data-pipeline.md)
  - [π0.5](../../wiki/entities/paper-pi05-open-world-vla.md)

## BibTeX

```bibtex
@article{wang2026emergenttransfer,
  title   = {When Does Legacy Data Start to Help? Emergent Transfer in Cross-Configuration Robot Learning},
  author  = {Wang, Tao and Hou, Hudson and Hu, Yingdong and Liu, Yufeng and Li, Qinghai
             and Jiang, Yingjie and Wang, Yingzhi and Ma, Cheng and Wang, Richard and Gao, Yang},
  journal = {arXiv preprint arXiv:2607.25593},
  year    = {2026}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-emergent-transfer-cross-config.md`](../../wiki/entities/paper-emergent-transfer-cross-config.md)
- 互链：[跨具身迁移 hub](../../wiki/overview/hub-cross-embodiment.md)、[跨具身策略迁移选型指南](../../wiki/queries/cross-embodiment-transfer-strategy.md)、[人形训练数据管线](../../wiki/queries/humanoid-training-data-pipeline.md)、[π0.5](../../wiki/entities/paper-pi05-open-world-vla.md)、[VLA](../../wiki/methods/vla.md)、[Behavior Cloning](../../wiki/methods/behavior-cloning.md)

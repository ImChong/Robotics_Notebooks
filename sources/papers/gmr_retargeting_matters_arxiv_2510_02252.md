# GMR：Retargeting Matters（arXiv:2510.02252）

> 来源归档（paper + 官方仓库核对）

- **类型：** paper
- **标题：** Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking
- **arXiv：** <https://arxiv.org/abs/2510.02252>
- **PDF：** <https://arxiv.org/pdf/2510.02252>
- **作者：** Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, C. Karen Liu（arXiv 摘要页）
- **提交：** 2025-10-02
- **代码：** <https://github.com/YanjieZe/GMR>（git ls-remote 可达，2026-11-05 HEAD `25f56db`；MIT；README 标版本 0.2.0）
- **论文网站：** <https://jaraujo98.github.io/retargeting_matters>（摘要页所列，未打开）
- **入库日期：** 2026-11-05；2026-11-07（全文通读更新）
- **核对范围：** 通读 arXiv:2510.02252 v1 全文 §I–§VI（引言、相关工作、三种基线与评测方法、GMR 五步流程、成功率与跟踪误差表、伪影分析、用户研究、首帧影响、结论与局限）；官方 README（支持机器人 / 数据格式表、安装、用法段首）。逐条成功率表的数值只读了汇总结论，未逐格核对；未运行代码，未读 DOC.md。
- **通读：** 已完成
- **线索来源：** [wechat_yuanxq_retargeting_general_tracking_2026-11-05.md](../blogs/wechat_yuanxq_retargeting_general_tracking_2026-11-05.md)

## arXiv 摘要要点（人工转述）

摘要称：作者系统评估重定向质量如何影响动作跟踪策略（在抑制过度奖励调参的条件下），指出脚滑、自穿透和物理不可行等重定向伪影常被留给 RL 策略去纠正；提出 GMR，并与 PHC、ProtoMotions 两个开源重定向器及 Unitree 的闭源高质量数据集比较；用 BeyondMimic 训练策略，在 LAFAN1 子集上称 GMR 在跟踪表现与对源动作的忠实度上持续优于现有开源方法，接近闭源基线。

## 命名与易混点

- 官方 README 同时挂了两个 arXiv 徽章：2505.02833 是 TWIST（Teleoperated Whole-Body Imitation System，见 [paper-twist](../../wiki/entities/paper-twist.md)），GMR 本体技术报告是 2510.02252。
- 仓库 `engineai-robotics/GMR` 等厂商分支见 `cn-os-*-gmr` 页，不是本条目。

## 对 wiki 的映射

- [motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)

## 全文核对要点（2026-11-07，均出自论文 / README）

- **问题与结论：** 在 BeyondMimic（几乎不调奖励）下，对 LAFAN1 的 21 段序列（5 s–2 min，仅脚接触的子集）比较 PHC、ProtoMotions、GMR 与 Unitree 官方重定向（闭源）。多数动作都能被跟踪，但**地面穿透、自穿透、关节值突变**三类伪影会显著降低鲁棒性；GMR 在跟踪成功率和用户研究的"忠实度"上优于另两种开源方法，接近 Unitree 数据（用户研究 N=20，Unitree 被评为更忠实但常难以与 GMR 区分）。
- **GMR 五步（§IV）：** ①人-机关键刚体匹配（可设位置 / 旋转权重）；②静止姿态下的笛卡尔空间朝向对齐（可加局部位置偏移，缓解 toed-in）；③**非均匀局部缩放** $p_b^{target}=\tfrac{h}{h_{ref}}s_b(p_b^{source}-p_{root}^{source})+\tfrac{h}{h_{ref}}s_{root}p_{root}^{source}$（先按身高得总体比例，再按刚体定制 $s_b$；**根平移必须用统一缩放，否则会引入脚滑**）；④第一阶段 IK：只含全部匹配刚体的朝向与末端（手 / 脚）位置，用 **Mink 微分 IK**（每步解 $\min_{\dot q}\|e(q)+J\dot q\|_W^2$，受关节限位），收敛阈值 0.001、最多 10 次迭代；⑤第二阶段：以第一阶段解为初值，加入全部关键刚体的位置与旋转约束并换一组权重微调。序列处理时逐帧、上一帧结果作初值；最后做高度修正（减去最低点）。
- **基线机理：** PHC 用 SMPL 拟合机器人形状 + Adam/Adadelta 梯度下降解 IK，不考虑接触，需额外抬高修正（出现过 30 cm 以上悬空）；ProtoMotions 用全局轴向缩放 + Mink。
- **评测数值（原文）：** 21 段里 11 段所有方法成功率均 >98%，3 段满分，其余 7 段差异大；sim / sim-dr / sim2sim 三种设置（100 / 4096 / 100 次）；失败案例对应伪影：PHC 的 Dance 1/2 地面穿透（最高 60 cm）、ProtoMotions 的 Run(stop&go) 腿部相交、GMR 的 Dance 5 腰部 roll 突跳（不到全数据集的 2%，因所有实验用同一组权重）。起始帧对成功率影响很大（表 III：同一策略仅改变参考动作的起始帧，sim2sim 成功率可从个位数 / 两位数百分比变到接近 100%；具体列对应关系以原文表格为准），建议起始位姿可安全到达、结束于稳定姿态。
- **局限（原文）：** 数据只来自 LAFAN1；只测 Unitree G1（受 BeyondMimic 代码库限制）；不含与环境 / 物体 / 其他机器人的交互。
- **README 补充：** 许可 MIT；README 同时挂 arXiv:2505.02833（TWIST）与 2510.02252 两枚徽章；支持机器人表列出 ID 0–18 已有数据格式支持（G1、H1、H2、Booster T1/K1、Fourier GR3 等）、ID 19–20 尚 TBD；输入格式含 SMPL-X（AMASS / OMOMO）、LAFAN1 BVH、FBX、Xsens / Nokov BVH、GVHMR 单目视频、PICO 流式。README 称"real-time"，论文未给出毫秒级耗时；ReActor 论文报告在 CPU 上重定向整个过滤后 AMASS 约需 5 小时。

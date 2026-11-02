# 黑箱策略传感器诊断修复（arXiv:2610.36852）

> 来源归档（paper）

- **标题：** When Black-Box Robot Policies Fail: Online Sensor Diagnosis and Repair with a Transparent Surrogate
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.36852>
- **PDF：** <https://arxiv.org/pdf/2610.36852>
- **作者：** Ziegler, Marc, Özen, Batu, Mentes, Sami, Waschulzik, Thomas, Knoll, Alois
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 黑盒机器人策略失效时：基于透明代理的在线传感器诊断与修复。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 冻结的黑盒运动策略遇到传感器故障时，无法通过访问策略内部结构定位和修复错误输入。
3. **周报所述创新点：** 用显式分离周期步态与传感器修正项的透明代理近似策略，再利用代理梯度诊断并修正输入；在 Go2 上将故障条件成功数由 259/640 提高到 382/640。
4. **arXiv 摘要（英文节选）：** Robot foundation models may generalize across tasks and embodiments, but their actions still depend on sensors that can fail. When the policy is frozen or inaccessible, can an external monitor diagnose and repair corrupted inputs without access to its internals? We test a structured surrogate that separates periodic gait from sensor-dependent corrections to guide input repair around a frozen locomotion policy.

**对 wiki 的映射**

- [paper-blackbox-policy-sensor-diagnosis-repair](../../wiki/entities/paper-blackbox-policy-sensor-diagnosis-repair.md)

# Blind Dexterous Grasping via Real2Sim2Real Tactile Policy Learning

> 来源归档（sun254667 Awesome 策展索引级）

- **列表：** [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real)
- **分组：** 42 Simulation Frameworks & Pipelines
- **编号：** 041/063
- **入库日期：** 2026-08-10
- **arXiv：** 2606.11767
- **出处：** arXiv 2026（v2，2026-06-11）
- **核对范围：** arXiv HTML v2 全文：§1–6 方法、20 物体实机评测、校准消融、局限、附录 A–G（传感器/指标/失败模式/PPO 专家/触觉编码器/扩散策略）；官方项目页（2026-11-23）。
- **通读：** 已完成
- **项目页：** <https://dex-blind-grasp.github.io/>
- **论文：** <https://arxiv.org/abs/2606.11767>
- **代码：** 未在清单中标注

- **Highlights（清单）：** Introduces a Real2Sim tactile calibration pipeline that constructs a contact-calibrated digital-twin simulator capable of reproducing real tactile signals. The deployed policy achieves a 27% real-world grasp success rate across all 20 objects without real-world grasping demonstrations or visual input
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2606-11767-blind-dexterous-grasping-via-real2sim2real-tacti.md`](../../wiki/entities/paper-sa-2606-11767-blind-dexterous-grasping-via-real2sim2real-tacti.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | 以项目页 / GitHub 实际链接为准；清单标注见上 |


## 一手论文核验（2026-11-23）

- **盲抓取的含义：** 无视觉输入；并非无触觉。真实平台为 xArm6 + 16-DoF LEAP Hand，四个指尖 TwinTac（32 路）与 12 个 FSR，共 44 路二值触觉。
- **Real2Sim：** 用 tapping/sliding 真机接触原语同步记录关节轨迹与二值触觉，在配对仿真器中重放，网格搜索接触阈值、指尖压力串扰参数以最小化事件流 L1 差异；标定使接触开始时间误差 426→96 ms，F1 64.2%→70.5%（论文 Table 2）。
- **策略：** 标定仿真中对象专属 PPO 专家采集成功轨迹；触觉编码器按手部 FK 把触觉激活与 3D 位置绑定，在仿真特权物体几何/位姿/接触监督下预训练；冻结编码器，用统一触觉条件 Diffusion Policy 从专家数据中学习动作块并部署真机。
- **真机结果：** 20 种物体、每种 5 次：已见 16/50、未见 11/50，合计 **27/100（27%）**。附录真机消融（关闭触觉标定的同条件比较）显示无预训练 6/100、有预训练 27/100；两种实验条件不可混说。
- **仿真消融：** 有预训练的已见 60.4%、未见 43.2% 为**仿真**成功率（Table 3），不能标作真机结果。
- **失败：** 73 次失败：空抓 15、抬起滑落 35、安全急停 23；触觉覆盖不全、滑移难检测且仍有接触动力学域差。
- **开源：** 2026-11-23 官方项目页 Code / Hardware 为 coming soon 占位链接，未提供可检查的官方训练或推理仓库，故无法制作源码级运行时序图。

## 官方项目资料

- [论文项目页归档](../sites/dex-blind-grasp.md)
- [原文 HTML v2](https://arxiv.org/html/2606.11767)

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2606-11767-blind-dexterous-grasping-via-real2sim2real-tacti.md`](../../wiki/entities/paper-sa-2606-11767-blind-dexterous-grasping-via-real2sim2real-tacti.md)
- 列表实体：[`wiki/entities/awesome-real2sim2real.md`](../../wiki/entities/awesome-real2sim2real.md)

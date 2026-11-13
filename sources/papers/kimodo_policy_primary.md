# Kimodo-Policy: Learning Kinematic Action Models for Humanoid Loco-Manipulation

- **类型：** 作者一手论文
- **作者：** Yunheng Wang、Jiaxi Zhang、Wei Lu、Yuetong Fang、Taowen Wang、Yixiao Feng、Shuning Zhang、Junzhe Xu、Yichi Wang、Zizhao Yuan、Kailun Huang、Lusong Li、Mingqian Zhou、Zecui Zeng、Renjing Xu
- **署名说明：** Yunheng Wang、Jiaxi Zhang、Wei Lu 为共同一作；Zecui Zeng、Renjing Xu 为共同通讯作者。
- **会议：** arXiv 2026 preprint
- **机构：** 香港科技大学（广州）（HKUST Guangzhou）；京东探索研究院（JD Explore Academy）；北京工业大学（BJUT）；剑桥大学（Cambridge）
- **论文：** https://arxiv.org/abs/2611.11554v1
- **原始版本：** arXiv:2611.11554（以所列论文链接版本为准）
- **项目页：** https://kimodo-policy.github.io/
- **代码：** https://github.com/kimodo-policy/kimodo-policy
- **核对范围：** arXiv:2611.11554v1 全文：§1–2 KAM 定义与相关研究、§3 Root/Body/Hand、可视 ControlNet、DINOv3/LLM2Vec、实测历史重规划与损失、§4 419h 多数据源训练/真机采集/追踪器部署、§5 HumanoidArena 七任务与 SIMPLE 八任务/真机八任务/权重冻结及数据和模型容量消融、§6 结果边界与未来研究；含图表及参考文献浏览。论文 HTML 无另附独立 Appendix。
- **通读：** 已完成

## 方法与应用

KAM 的预训练对象是运动学序列，而不是视觉语义；冻结的 Kimodo Root/Body 去噪骨干提供肢体协同先验。Trainable ControlNet 经双路径视觉融合注入根部与身体阶段，HandNet 根据未来身体、图像和当前双手状态生成开合命令。每次使用机器人实测动作历史而非仅重放预定动作。

## 实验与结论

论文报告 HumanoidArena 七任务联合训练平均 72.14%；模型 678M 参数、92M 可训练；真机八任务 60%（48/80），对照移除 419 小时交互数据预训练为 36/80。仿真三项 HOI 消融 18.33/31.67/51.67% 不能和七任务总分混算。

## 局限 / 对 wiki 的映射

文中 1.67 秒预测、0.5 秒执行前缀只属于本策略配置；部署依赖视觉、机器人状态与 SONIC，预训练消融与真实交互数据来源不可混算。

- [Kimodo-Policy：运动动作模型的视觉闭环移动操作](../../wiki/entities/paper-kimodo-policy.md)

## 原文复核摘录（正文证据）
- **图像到动作：** 保留冻结 Kimodo-G1 Root/Body 与 DINOv3，使用视觉 ControlNet 两路融合；训练 HandNet 的手部梯度不回传给 Motion Backbone；训练损失 root/body/hand/transition 权重 4/1/1/0.2。
- **时间与部署：** 30 Hz 历史 100 帧、预测 50 帧（1.67 秒）、执行 15 帧（0.5 秒）；SONIC 50 Hz。真机启用 RTC，仿真评测关 RTC；SIMPLE 可执行 15 或 50 帧（论文 §4.2）。
- **结果区别：** HumanoidArena 七任务多任务 72.14%，但单任务直接微调 69.76%，额外预训练后 67.14%；SIMPLE 8 任务 75.83%（182/240）对照 Ψ0 75.42%（181/240）。真机 36/80→48/80（45%→60%）。三项 HOI 消融 18.33/31.67/51.67% 与七任务总分不同。
- **局限：** §6 明确当前只适配 SONIC tracker，跨 tracker 未验证；独立 419h robot-interaction pretraining 的收益不可与 Kimodo-G1 运动骨干预训练混淆。
- **参考：** §3–§6 含全部主要结果表与 Figure 14 消融；[论文原文](https://arxiv.org/html/2611.11554v1)。

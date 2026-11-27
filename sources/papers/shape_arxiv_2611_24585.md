# SHAPE · 第一视角人类数据的灵巧操作学习

- **类型：** paper
- **原文：** [arXiv:2611.24585v1](https://arxiv.org/abs/2611.24585v1) · [全文 HTML](https://arxiv.org/html/2611.24585v1) · [PDF](https://arxiv.org/pdf/2611.24585v1)
- **发布日期：** 2026-11-18
- **作者：** Yaping Li、Zhihong Liu、Jiacheng Li、Yi Yang、Wenchao Yue、Mingzhe Ye、Xiangcheng Liu、Yiyang Chen、Haotian He、Qi Zhang、Haonan Wang、Dahua Lin、Zhijie Deng
- **机构：** Sharpa Robotics、香港中文大学、上海交通大学–Sharpa 具身智能联合实验室（按 v1 论文首页）
- **项目页：** https://li-yaping.github.io/SHAPE/
- **核对范围：** v1 PDF 28 页，正文 §1–7 与附录 A–G，包括数据配比、动作参数、真机分阶段计分、统计、姿态偏差分析、梯度匹配对照及 GR-1 仿真协议。
- **通读：** 已完成

## 摘录 1：监督覆盖不同，两条连续流共享专家 (§3；附录 A)

人第一视角预训练样本提供重定向末端动作和 FAST token，真实遥操作还监督关节流。两种来源可影响同一个骨干和动作专家，但不能把人视频当作真实关节标签。FAST 以归一化的 EEF chunk 产生离散标签，骨干预测它们；动作专家不能读取这些标签，部署省去该分支。

## 摘录 2：按示教域分开报告真机成绩 (§5.1–5.3；附录 B–D)

六项真机任务每项 20 轮，USB 和 Type-C 部分进度参与计分。SHAPE 关节流 0.80，π0.5 0.63，GR00T N1.7 0.51；严格完成分别为 76、61、51 / 120。三项不使用任务专用真机示教的任务由人类样本微调：完整 SHAPE 在同一 checkpoint 上 EEF 21/60、关节 0/60。该实验保留先前的通用机器人预训练，且人类微调给关节流的只有重定向标签，不是真机关节真值。

## 摘录 3：结果对应协议，而不是动作空间的普适排名 (§5.4、§6；附录 E–G)

GR-1 在 RoboCasa 已使用 20,802 条同任务遥操作训练样本，共 24 项、每项 50 个种子、三个 checkpoint；作者报告 72.2%，不是第二台实机的零样本验证。梯度匹配连续头在真机人到机器人协议中为 18.3%，对比 FAST 的 35.0%，分层精确检验 p=0.055；不宜宣称离散监督已在这一对比中显著胜出。附录 E.5 的 FAST 直接解码在机器人测试窗口约 15% 无效，不能当作已部署的连续控制头。

对应 [SHAPE 实体](../../wiki/entities/paper-shape-egocentric-dexterous-manipulation.md)。

# Humanoid-VLA

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 121/161）

- **标题：** Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2502.14795>
- **Loco-Manip 161 分类：** 06 特殊任务、接触规划与视觉闭环
- **作者：** Pengxiang Ding, Jianfei Ma, Xinyang Tong 等（西湖大学 Milab、浙江大学、Westlake Robotics；通信作者 Donglin Wang）
- **机构：** 西湖大学 / 浙江大学 / Westlake Robotics
- **项目页：** 见 arXiv:2502.14795
- **发表日期：** 2025年2月21日（arXiv:2502.14795v2）
- **入库日期：** 2026-06-26
- **通读：** 已完成
- **核对范围：** 已下载并通读 `arXiv:2502.14795v2` PDF（Abstract、§1 引言、§3 方法、§4.1 HumanML3D/Humanoid-S/IsaacGym 定量表、§4.2 Unitree G1 八项真机任务成功率表与 §5 结论）
- **一句话说明：** Humanoid-VLA 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、全身控制器/WBC/MPC预测全身轨迹/动作序列。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 06 特殊任务、接触规划与视觉闭环，编号 **121/161**。
- **算法实现总结（公众号）：** Humanoid-VLA 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、全身控制器/WBC/MPC预测全身轨迹/动作序列。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 对 wiki 的映射

- [paper-loco-manip-161-121-humanoid-vla](../../wiki/entities/paper-loco-manip-161-121-humanoid-vla.md)
- [loco-manip-161-category-06-contact-tasks](../../wiki/overview/loco-manip-161-category-06-contact-tasks.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)

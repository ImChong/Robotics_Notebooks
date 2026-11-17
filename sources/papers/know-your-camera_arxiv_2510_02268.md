# Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning（一手论文阅读归档）

- **类型：** paper / 机器人研究获奖论文
- **标题：** Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning
- **中文题名：** Do You Know Where Your Camera Is?：用相机几何条件实现视角鲁棒策略学习
- **arXiv：** <https://arxiv.org/abs/2510.02268>
- **原文 HTML：** <https://arxiv.org/html/2510.02268>
- **PDF：** <https://arxiv.org/pdf/2510.02268>
- **版本：** v1（2025-10-02）
- **会议奖项线索：** ICRA 2026 Best Paper Award on Robot Learning（奖项由二手综述提出，与一手项目资源交叉核对）
- **项目页：** <https://ripl.github.io/know_your_camera/>
- **代码：** <https://github.com/ripl/CamPoseOpensource>
- **核对范围：** arXiv HTML v1：摘要、§I–V、Plücker 编码、六项 RoboSuite/ManiSkill 设置、表 I、图 6–8、真实 UR5 协议及结论局限；原始公开仓库 README 与复现 CLI 源码。
- **通读：** 已完成
- **开源核验：** 已开源：训练代码、六项仿真任务、示范和论文逐图实验运行清单
- **入库日期：** 2026-11-17

## 要点摘录与 wiki 映射

1. PLücker ray-map 用方向 d 和矩 C×d，输入比仅相机平移/旋转数字更像素对齐。
2. 静态背景可提供相机捷径，随机场景是重要对照。
3. 作者区分模拟 1500 rollouts 和真实每设置 21 次 rollouts，不能混报。

- **技术机制：** 由 K 内参和 camera-to-world 外参为每像素反投影得到世界方向 d；以相机中心 C 构造 Plücker 矩 m=C×d，拼接六维 (d,m) 射线图，与 RGB 一起输入 ACT、Diffusion Policy、SmolVLA；裁剪增强时必须同步裁剪 RGB 与 ray-map。
- **评测证据：** 六项 RoboSuite/ManiSkill 仿真任务与三项真实 UR5 操作；表 I 中 ACT 的 Lift 成功率 **33.6%→60.6%**，DP 的 Pick Place Can **23.1%→39.3%**，SmolVLA 的 Lift **19.6%→54.4%**（均为对应模拟设置）。仿真 ACT/DP 各设置三种种子×10 checkpoint×50 场景 = 1500 rollout。
- **局限：** 准确内外参是前提，遮挡、动态或无纹理环境的姿态估计误差会传递到控制；论文主要考察外参变化，跨不同内参相机仍待扩展；不能把模拟增益直接当作真实机器人的全任务成功率。
- **对 wiki 的映射：** [Do You Know Where Your Camera Is?：用相机几何条件实现视角鲁棒策略学习](../../wiki/entities/paper-know-your-camera.md)（每篇论文只创建一个独立详情节点）
- **二手索引：** [2026 四大顶会 12 篇获奖论文](../blogs/wechat_shenlan_four_robotics_conferences_12_awards_2026_11_17.md)（仅奖项线索，非技术证据）

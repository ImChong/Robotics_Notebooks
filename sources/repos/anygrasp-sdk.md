# AnyGrasp SDK（通用抓取感知与跟踪）

- **标题**: AnyGrasp SDK — grasp detection & tracking
- **代码**: https://github.com/graspnet/anygrasp_sdk
- **论文（预印本）**: https://arxiv.org/abs/2212.08333（IEEE T-RO 2023 正式版见官方引用）
- **项目页 / Demo 汇总**: https://graspnet.net/anygrasp.html
- **数据集与基准（GraspNet 系列）**: https://graspnet.net/datasets.html
- **Python 数据接口**: https://github.com/graspnet/graspnetAPI
- **网络结构参考实现（非 SDK 权重）**: https://github.com/graspnet/graspnet-baseline
- **GSNet 非官方实现（学习用）**: https://github.com/graspnet/graspness_unofficial
- **类型**: repo / commercial-license-binary / perception
- **团队**: 上海交通大学 MVIG（Machine Vision and Intelligence Group）等
- **收录日期**: 2026-05-13

## 一句话摘要

面向**平行夹爪**的 **7-DoF 稠密抓取位姿**估计与**跨帧关联跟踪**：单前向 pass 从单目深度点云生成大量候选，强调对真实深度噪声鲁棒、时空连续性，以及质心（COG）相关的稳定度建模；**推理库以预编译二进制 + 机器 License 发放**（非全开源权重）。

## 为何值得保留

- **工程闭环**：与「只发论文」相比，提供可申请的 SDK、检测与跟踪 demo，便于接入 bin picking、动态抓取原型。
- **方法谱系清晰**：几何模块建立在 **GSNet / Graspness** 一脉的端到端稠密预测上，并显式增加 **时序关联模块** 与 **COG 稳定分数**、障碍感知等设计。
- **数据叙事**：论文强调在**小规模真实物体集（约 144 个物体、268 场景）**上训练即可在多种挑战场景接近人类水平，与「堆模拟物体数量」路线形成对照讨论素材。

## 技术要点（来自论文摘要、正文与官方 README 的公开描述）

1. **表示**: 抓取配置为 **SE(3) 位姿 + 夹爪开度**，即论文中的 **7-DoF** 平行夹爪参数化。
2. **空间稠密**: 几何处理模块对**部分视点云**一次前向预测大量候选；相对采样–再评估路线，强调**稠密 + 速度**折中。
3. **时间连续**: **Temporal association** 在相邻观测的稠密抓取之间建 **many-to-many** 关联，使同一目标在物体坐标系下跨帧一致（动态跟踪 / 连续更新抓取目标）。
4. **稳定与可执行性**: 引入 **COG 相关稳定分数**（假设均匀密度刚体近似质心）；网络侧 **障碍感知** 将无预置空间的抓取质量置零，减少显式碰撞检测负担（与纯几何管线对比）。
5. **训练数据**: 主要继承 **GraspNet-1Billion** 训练集并扩展真实场景；使用 **解析 antipodal 分数** 等自动生成稠密标签，另用**多视点相邻图像**构造关联监督，弥补「真实动态抓取数据集稀缺」。
6. **SDK 形态（README）**: 依赖 **PyTorch**、**MinkowskiEngine**（维护者 fork 安装说明）、`pointnet2` 扩展；提供 `dense_grasp`、`apply_object_mask`、`collision_detection` 等开关；**License** 需填表申请（IP 限制），实现细节可参考 **graspnet-baseline** 与第三方 GSNet。

## 对 Wiki 的映射

- **wiki/entities/anygrasp.md**：抓取感知实体页（方法边界、管线、局限与选型）。
- **wiki/tasks/manipulation.md**：在「视觉感知 → 抓取规划」链路上互链。
- **references/repos/manipulation-perception.md**：操作感知向开源导航入口。

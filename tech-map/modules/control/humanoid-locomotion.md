# 人形 Locomotion（双足行走）

人形双足行走要同时解决三件事：**怎么迈步**（步态与落脚点）、**怎么不摔**（平衡与扰动恢复）、**地形变了怎么办**（感知与自适应）。它是本站的主线任务之一，传统控制（MPC + WBC）与强化学习两条路线都在这里正面相遇。

**站内入口：** [Locomotion 任务页](../../../wiki/tasks/locomotion.md) · [强化学习 Locomotion 纵深路线](../../../roadmap/depth-rl-locomotion.md) · [常见失败模式](../../../wiki/queries/locomotion-failure-modes.md) · [奖励设计指南](../../../wiki/queries/locomotion-reward-design-guide.md)

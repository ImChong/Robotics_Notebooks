# 人形运控策略观测输入分类 FAQ 摘录（维护者整理）

- **类型**：`personal`（对话/答疑整理，非正式出版物）
- **日期**：2026-07-28
- **用途**：为 [人形机器人运控策略的观测输入](../../wiki/concepts/humanoid-policy-observation-inputs.md) 提供可追溯的编译来源说明；正文以 wiki 页为准，本文件不重复展开技术细节。
- **综合依据**：本库已 ingest 的 [privileged_training.md](../papers/privileged_training.md)、[state_estimation.md](../papers/state_estimation.md)、[rma_arxiv_2107_04034.md](../papers/rma_arxiv_2107_04034.md) 与 [perceptive_locomotion_representation_essence.md](./perceptive_locomotion_representation_essence.md)，以及 legged_gym / Isaac Lab 系开源训练栈的公开 observation 配置。

## 对话要点（溯源用）

- 主流人形/腿式运控策略（RL / IL / 跟踪系）的输入可按「部署是否可得」切成五类：本体感知、指令与参考、历史上下文、外部感知、特权信息（仅训练）。
- 每类的关键工程问题不是「有什么量」，而是「这个量在真机上怎么拿到」：直读（编码器/IMU）、滤波估计（EKF）、学习估计（RMA / CENet）、感知管线（高程图/深度）、上层给定（指令/参考）。
- 基座线速度是最典型的「仿真直读、真机须估」量；yaw 角不可全局观测，主流做法是给 projected gravity 而不是欧拉角。

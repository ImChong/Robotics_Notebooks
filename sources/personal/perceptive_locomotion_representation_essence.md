# 感知 Locomotion：地形 Latent 与 Teacher–Student 蒸馏本质 FAQ（维护者整理）

- **类型**：`personal`（对话/答疑整理，非正式出版物）
- **日期**：2026-06-14
- **原始对话**（仅作溯源，不在 wiki 建独立节点）：
  - 感知模型本质分析：<https://chatgpt.com/share/6a2ed638-41a4-83ea-b5bd-8b9d5661fc96>
- **用途**：为 [地形 Latent 表征](../../wiki/concepts/terrain-latent-representation.md)、[特权信息训练](../../wiki/concepts/privileged-training.md)、[地形适应](../../wiki/concepts/terrain-adaptation.md) 提供可追溯编译来源；正文以 wiki 页为准。

## 对话要点 1：Latent ≠ 高度图

- 深度图经 CNN/Transformer 编码后得到的 **terrain latent / terrain embedding**，在主流感知行走工作中 **通常不是** 可读的 64×64 高度栅格。
- 信息流：`真实地形 → 深度相机 → Depth Image → Encoder → Terrain Latent（如 128 维）→ Policy → Action`。
- Latent 是 **周围地形的压缩摘要**（台阶高度、坡度、障碍位置、可落脚区域），人类无法逐维解读，类似语言模型的 sentence embedding。
- 少数工作显式重建 Height Map 再喂 policy；更常见的是 **Height Map → Encoder → 低维 latent** 或直接 **Depth → latent**，因 RL 偏好 64–256 维紧凑输入。
- 代表线：Perceptive-BFM、PHP-Parkour、Walk These Ways、Extreme Parkour 等多属后者。

## 对话要点 2：Teacher–Student 蒸馏本质

- 蒸馏的不是网络结构或参数，而是 **Teacher 在大量场景下的决策经验**。
- 无 Teacher：RL 靠探索 + 稀疏奖励慢慢找答案；有 Teacher：每个场景 `obs → Teacher → action` 构成 **有标签数据集**，Student 做监督学习（动作回归 / BC），**无 PPO、无探索**。
- **最本质一句话**：Teacher 把 **未知答案的 RL 问题** 转成 **已知答案的监督学习问题**。
- Teacher 通常 **也能看到 Student 的传感器**（如深度图），并 **额外拥有特权信息**（高度图、未来地形、接触状态、仿真真值）；原则：`obs_teacher ⊇ obs_student`。
- 极端情况：Teacher 仅高度图、Student 仅深度——Student 需从深度 **隐式恢复** 地形信息再输出动作（Walk These Ways 的 adaptation 思路）。

## 对 wiki 的映射

| 要点 | 目标页 |
|------|--------|
| Latent vs Height Map、感知信息流 | `wiki/concepts/terrain-latent-representation.md`（新建） |
| 蒸馏 = RL→监督、Teacher 观测 ⊇ Student | `wiki/concepts/privileged-training.md`（增补「蒸馏本质」节） |
| 楼梯/障碍感知行走索引 | `wiki/tasks/stair-obstacle-perceptive-locomotion.md` |
| Perceptive-BFM / Extreme Parkour 等实体 | 已有实体页，补 `related` 链回新概念页 |

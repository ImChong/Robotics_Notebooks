# CoRE 论文来源归档

## 元数据

- **题目：** CoRE: Learning Collaboration-Role Experts for Decentralized Collaborative Manipulation with One Policy
- **方法：** CoRE（Collaboration-Role Experts）
- **作者：** Yanan Zhou, Zhaoyan Qian, Zihao Li, Mingyuan Ba, Ranpeng Qiu, Weiming Zhi
- **单位：** The University of Sydney 计算机学院（Yanan Zhou、Zhaoyan Qian、Zihao Li、Mingyuan Ba、Weiming Zhi，据项目页）；Zhejiang University of Technology（Ranpeng Qiu）；PAIR Lab（Physical AI & Robotics，Sydney）；通讯作者 Weiming Zhi
- **预印本：** https://arxiv.org/abs/2610.07752
- **HTML：** https://arxiv.org/html/2610.07752v1
- **版本：** v1，2026-10-06，cs.RO，CC BY 4.0，备注 “8 pages, 10 figures”
- **项目页：** https://aus.bot/research/core/ （归档见 [core-collab-role-experts-project.md](../sites/core-collab-role-experts-project.md)）
- **代码：** 项目页标 “Coming soon”，截至 2026-10-12 无仓库链接。

## 论文要解决的问题

去中心化协同操作：每个机器人运行同一网络，只用自己的视觉观测与本体感觉，没有任务提示、身份标签或机器人间通信。难点是在共享参数里学到互补的团队行为，并仅凭局部信息选择自己的角色动作。

## 方法与训练信息

1. 感知：StereoPatch 在共享 patch 网格上融合 RGB 特征（DINOv3）与配准深度特征（DeFM），得视觉 token X_t，加本体感觉 q_t。
2. 专家：动作解码器每层含 E 个协作角色专家 + 共享交叉注意力；每个专家用低秩投影对视觉 token 做 query-conditioned 交叉注意力，每个预测位置有可学习动作查询 u_k。
3. 局部路由：MLP 以 X_t 均值、q_t 编码、u_k 为输入打分，softmax 后 top-K 重归一化，修正量加到前馈块之前。
4. 动作–专家对齐损失 L_AEA：依次强制每个专家，用逐位置误差 ℓ=(1/d_a)‖a*−â‖² 做 stop-gradient 的 softmax(−ℓ/τ) 目标，对路由概率求 KL，无需角色标签。
5. 动作块 H=100，Transformer 动作块回归器 + CVAE 隐变量（部署置零），时间集成系数 0.01；超参：4 个 rank-32 专家，top-2，7 层解码器，λ_CVAE=1e-3，λ_AEA=0.05。
6. 训练：仿真 120,000 次 AdamW 更新、批 40，评测每任务 100 seed；真机 80,000 次更新、批 32、640×320 RGB。

## 论文报告的结果

- 仿真 13 个任务（RoboFactory 5、DuoBench 4、BiCoord 4；2–4 机器人）。去中心化方法平均成功率：CoRE 在 DuoBench 47.6%、BiCoord 75.5%、RoboFactory 84.2%；CHORUS 43.5 / 26.5 / 58.4；CLS-DP 40.7 / 17.3 / 73.8；GauDP 27.4 / 14.0 / 20.2；独立回归 25.0 / 21.0 / 14.0。中心化回归（联合观测）在 RoboFactory 达 90.4%。
- 演示合并（RoboFactory）：分任务回归 53.6，全任务 RGB 回归 40.4，CoRE(RGB) 72.4，完整 CoRE 84.2。
- 消融：RGB 回归 40.4 → Stereo 回归 56.6 → +MoE 62.4 → +ARCA 64.2 → CoRE(RGB) 72.4 → 完整 84.2；(E,K) 稀疏度：(4,2) 84.2，(2,2) 80.2，(4,1) 62.6，(4,4) 58.8，(8,2) 83.4；对齐损失使 LPD 任务由 15% 升至 94%。
- 真机：两台 Piper 臂 + AR0234 立体传感，5 个任务各 30 条示范；每方法每任务 30 次，CoRE 91.3%，CHORUS 56.0%，CLS-DP 35.3%；Bridge 装配 CoRE 25/30，CHORUS 7/30。扰动：Box 暂停 1.0 s，CoRE 22/30 vs CHORUS 4/30；Pan 0.25× 速度，17/30 vs 1/30。

## 复现与解释边界

- 无公开代码；论文无独立局限节。结果显示 Take Photo 仍弱（完整 CoRE 29%），DuoBench 平均仅 47.6%，真机仅两台机器人 5 个任务，每任务单一扰动类型。
- 中心化回归在 RoboFactory 上仍高于 CoRE（90.4 vs 84.2）。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-core-collaboration-role-experts.md
- 概念入口：wiki/concepts/mixture-of-experts.md
- 任务入口：wiki/tasks/bimanual-manipulation.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.07752>
- arXiv HTML：<https://arxiv.org/html/2610.07752v1>
- 项目页：<https://aus.bot/research/core/>

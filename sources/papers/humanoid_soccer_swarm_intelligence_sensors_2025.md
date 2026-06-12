# Swarm Intelligence for Collaborative Play in Humanoid Soccer Teams

> 来源归档（ingest · 人形机器人群控 / 多机协作）

- **标题：** Swarm Intelligence for Collaborative Play in Humanoid Soccer Teams
- **类型：** paper
- **期刊：** Sensors 2025, 25(11), 3496
- **DOI：** <https://doi.org/10.3390/s25113496>
- **PDF：** <https://www.mdpi.com/1424-8220/25/11/3496/pdf>
- **作者：** Farzad Nadiri, Ahmad B. Rad（Simon Fraser University, Autonomous and Intelligent Systems Laboratory）
- **特刊：** Robot Swarm Collaboration in the Unstructured Environment
- **入库日期：** 2026-06-12
- **一句话说明：** 面向 **人形足球机器人队** 的 **去中心化群智能框架**：轻量 UDP 通信 + **ACO 蚁群角色分配** + **Reynolds  flocking 编队** + 轻量 RL 自适应与故障恢复；Webots 4v4 仿真相对集中式基线 **进球 +25–40%**、控球 **+8–10%**，角色重分配 **2.3±0.4 s**（集中式 **5.1±0.6 s**）。

## 核心论文摘录（MVP）

### 1) 问题与动机（§1）

- **链接：** <https://doi.org/10.3390/s25113496>
- **摘录要点：** 人形足球在 **部分可观测、高动态、对抗** 环境下需要实时协作；传统 **集中式**（全局视觉/中央服务器）存在通信瓶颈、延迟、单点故障与扩展性差。SPL 近年 **WiFi 包率大幅下降**（见 arXiv:2401.15026），推动 **分布式协调** 成为刚需。本文将 **蚁群优化（ACO）** 与 **Reynolds flocking** 引入 **双足人形足球**——此前 swarm 研究多集中在轮式/简单任务。
- **对 wiki 的映射：**
  - [人形多机协调](../../wiki/concepts/humanoid-multi-robot-coordination.md)
  - [Humanoid Soccer](../../wiki/tasks/humanoid-soccer.md)

### 2) 四层架构（Figure 2）

- **摘录要点：**
  - **(A) 机载传感：** RGB 相机 + IMU/编码器，局部估计球、队友、对手与自姿态。
  - **(C) 通信模块：** **UDP** 广播位置、速度、角色（门将/后卫/中场/前锋）、任务优先级、电量/健康、可选意图字段；**RLE 压缩** + 拥塞退避；丢包时本地估计/fallback。
  - **(D) 群智能层：**
    - **D1 ACO 角色分配：** 每机器人维护各角色信息素 $\tau_i[r]$；适应度含距球距离、踢球能力、电量；**平衡惩罚** $\text{penalty}(r)=\gamma\cdot\max(0,n(r)-1)$（$\gamma=0.2$）防角色扎堆；周期 **100 ms**。
    - **D2 Flocking 编队：** cohesion / alignment / separation，按进攻/防守 **角色加权** 调制。
    - **D3 自适应行为：** 监测网络与机器人可用性，丢员/断联时触发 fallback 重分配。
  - **(E) 运动控制：** 将高层角色/站位转为双足步态与踢球指令。
- **对 wiki 的映射：**
  - [paper-humanoid-soccer-swarm-intelligence](../../wiki/entities/paper-humanoid-soccer-swarm-intelligence.md) — Mermaid 架构图

### 3) 实现与仿真平台（§2）

- **摘录要点：** 先 **PyGame 2D** 原型验证追球/角色/编队，再迁移 **Webots R2025a** 3D 人形动力学；控制周期约 **96 ms**（16 ms 物理步 × 6）；决策循环在 i5-8265U 上 **4.7±0.3 ms**（max 6.1 ms）。ACO 参数网格搜索：**α=1.0, β=2.0, ρ=0.5**；场地离散 **90×60 格、0.1 m**。
- **对 wiki 的映射：**
  - [MARL](../../wiki/methods/marl.md) — 与离线 MARL 对比：本文强调 **轻量本地交互、亚秒级在线重分配**

### 4) 主实验结果（§3，4v4 Webots）

- **摘录要点：**
  - **进球：** ACO+flocking 相对集中式/静态角色 **+25–40%**；控球 **+8–10%**；传球成功率 **+15–25%**。
  - **角色重分配：** **2.3±0.4 s** vs 集中式 **5.1±0.6 s**；球权转换后 **2–3 s** 完成重分配。
  - **鲁棒性：** 人工 **20% 丢包** 或临时禁用 1 台机器人，控球/进球仅降约 **8–10%**；集中式降幅更大。
  - **扩展场景（Table 3）：** 4v3 人少、**>50% 丢包**、角球/边线开球——相对标准 4v4（7.33 球、49.7% 控球）降幅 **<15%**。
- **局限：** 截至发表主要为 **仿真**；真机噪声、不平场地、参数调优（信息素蒸发、 flocking 距离）待验证；KidSize **4 机** 规则下验证，作者声称算法可线性扩展到 5–8 机。
- **对 wiki 的映射：**
  - [paper-humanoid-soccer-swarm-intelligence](../../wiki/entities/paper-humanoid-soccer-swarm-intelligence.md) — 实验表与局限

## 对 wiki 的映射（汇总）

- [paper-humanoid-soccer-swarm-intelligence.md](../../wiki/entities/paper-humanoid-soccer-swarm-intelligence.md) — 主沉淀页
- [humanoid-multi-robot-coordination.md](../../wiki/concepts/humanoid-multi-robot-coordination.md) — 去中心化群控范式
- 交叉更新：[humanoid-soccer.md](../../wiki/tasks/humanoid-soccer.md)、[marl.md](../../wiki/methods/marl.md)

## 引用（BibTeX）

```bibtex
@article{nadiri2025swarm,
  title   = {Swarm Intelligence for Collaborative Play in Humanoid Soccer Teams},
  author  = {Nadiri, Farzad and Rad, Ahmad B.},
  journal = {Sensors},
  volume  = {25},
  number  = {11},
  pages   = {3496},
  year    = {2025},
  doi     = {10.3390/s25113496}
}
```

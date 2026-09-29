# RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI

> 来源归档（ingest）

- **标题：** RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI
- **类型：** paper（系统 / Technical Report）
- **arXiv：** <https://arxiv.org/abs/2602.07837>
- **PDF：** <https://arxiv.org/pdf/2602.07837>
- **机构：** 清华大学、无问芯穹（Infinigence AI）、北京理工大学、浙江大学、中关村学院、跨步智能（Striding.AI）、上海 AI 实验室等；通讯 **Chao Yu**（yuchao@sz.tsinghua.edu.cn）、**Yu Wang**（yu-wang@mail.tsinghua.edu.cn）
- **会议：** **RSS 2026**（RLinf README 公告；与 RLinf-VLA 同批接收）
- **代码：** <https://github.com/RLinf/RLinf>（论文 Abstract 与 README 指向同一开源实现）
- **文档：** <https://rlinf.readthedocs.io/en/latest/rst_source/resources/publications/rlinf_user.html>
- **入库日期：** 2026-09-29
- **一句话说明：** **USER**（Unified and extensible SystEm for real-world online policy leaRning）把 **机器人与 GPU 同为可调度硬件**（HAL），用 **EasyTier 隧道 + 分布式 data channel + SM 预算 NCCL 同步** 支撑云–边 VLA 训练，并以 **全异步 env/rollout/actor + 持久化 cache-aware buffer** 承载 CNN/Flow/π₀ 上的 SAC、RLPD、SAC-Flow、HG-DAgger 等真机在线学习。

## 核心摘录（面向 wiki 编译）

### 1) 真机在线学习是系统问题

- **链接：** arXiv §I Introduction
- **摘录要点：**
  - 仿真可加速/重置/复制；真机 **实时、异构、长程、易中断**，瓶颈常在 **数据采集** 而非算力。
  - 机器人常被当作「外部环境」，难以与加速器 **联合调度**；云–边 VLA 带来 **跨域带宽不对称**。
  - 同步 pipeline 在真机上 **级联 stall**；Reverb/Flashbax 等 **内存型 buffer** 难支撑长程视觉数据与 crash recovery。
- **对 wiki 的映射：**
  - [RLinf-USER](../../wiki/entities/paper-rlinf-user.md) — 问题定义与 SERL/SOP/Qt-Opt 对照表（Tab. I）
  - [在线 vs 离线 RL](../../wiki/comparisons/online-vs-offline-rl.md) — 真机在线范式坐标

### 2) HAL：机器人与 GPU 同级调度

- **链接：** arXiv §III-A；Figure 2–3
- **摘录要点：**
  - **Hardware unit** = 单 GPU 或 **单台物理机器人**（可捆绑相机、SpaceMouse）。
  - 典型三类节点：**rollout（GPU 推理）**、**robot/env（CPU 边缘执行）**、**training（大算力集中训练）**。
  - **Rank-based placement** 统一绑定进程到 GPU 或 robot endpoint；同一 Job 可异构放置以权衡 **权重同步 vs env–rollout 通信**。
- **对 wiki 的映射：**
  - [RLinf-USER](../../wiki/entities/paper-rlinf-user.md) — HAL 与多机/异构实验（§V-B）
  - [RLark](../../wiki/entities/rlark.md) — 同生态 **跨集群编排** 的另一产品化路径（kcp/CRD；USER 论文栈为 Ray + EasyTier）

### 3) 自适应通信平面

- **链接：** arXiv §III-B；Tab. IV–V
- **摘录要点：**
  - **EasyTier UDP 隧道** 扁平化 NAT/厂区 VLAN；控制面 **Ray**，数据面 TCP rendezvous，流量绑隧道网卡。
  - **Distributed data channel**：按 robot ID 分片 FIFO，本地化 edge 流量；跨域 episode 生成 **~3×** 加速（21.98 s vs 69.27 s/episode，Tab. IV）。
  - **SM-budgeted NCCL**：限制 weight sync 占用 SM，避免拖慢 rollout 延迟。
- **对 wiki 的映射：**
  - [RLinf-USER](../../wiki/entities/paper-rlinf-user.md) — 通信消融与 π₀ HG-DAgger 异步加速（Tab. V）

### 4) 全异步学习与持久 buffer

- **链接：** arXiv §IV；Figure 5–7
- **摘录要点：**
  - **Env → rollout → 异步写 buffer**；**actor 异步采样**；权重 **周期性** 同步回 rollout，机器人 **不等待** 优化步。
  - Buffer：**磁盘持久 + 轻量索引**（policy version、timestamp、episode ID）；内存 **FIFO cache**，evict 后可从盘重载。
  - 策略：CNN / Flow / **π₀·π₀.₅ VLA**；算法：**SAC、RLPD、SAC-Flow、HG-DAgger**；奖励：规则 / 人类 / 学习 reward model。
- **对 wiki 的映射：**
  - [STEAM](../../wiki/entities/paper-steam-advantage-modeling.md) — 同 RLinf 仓 **离线** advantage 管线对照
  - [LeRobot](../../wiki/entities/lerobot.md) — VLA/数据格式生态

### 5) 真机实验摘要

- **链接：** arXiv §V；Tab. II–III
- **摘录要点：**
  - 五任务：Peg Insertion、Charger、Cap Tightening、Pick-and-Place、Table Clean-up（Franka + RealSense）。
  - RLPD/SAC/SAC-Flow 在 Peg/Charger 等 **~2000 s 墙钟** 近满分；π₀ **HG-DAgger** Pick-and-Place **~30 min、~200 online samples** 达 **96%** 成功率叙述。
  - π₀ 在线微调：**Pick-and-Place 39/60→58/60**；**Table Clean-up 9/20→16/20**（Tab. III）。
  - **四 Franka 跨 3 km 两站点** 并行 peg-insertion 与单机收敛相当；**Franka + ARX** 异构联合 SAC 收敛（~2 h）。
  - 相对 SERL 等：**真机 + 多机 + 异构 + 大模型 + 开源** 五维 Tab. I 唯一全开（USER）。
- **对 wiki 的映射：**
  - [RLinf-USER](../../wiki/entities/paper-rlinf-user.md) — 实验与结论
  - [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md) — 「搭集群 / 真机 RL 基建」入口

## 为何值得保留

- **RSS 2026 + arXiv 2602.07837** 给出 RLinf 生态在 **真机在线** 方向的 **系统论文锚点**，与 STEAM（离线 advantage）、RLark（K8s 跨集群）形成 **算法–系统–编排** 三角索引。
- 开源实现与 [RLinf 真机示例库](https://rlinf.readthedocs.io/en/latest/rst_source/examples/index.html)（Franka、HG-DAgger、ZED 等）可直接挂接，避免 wiki 只停留在仿真 RL。

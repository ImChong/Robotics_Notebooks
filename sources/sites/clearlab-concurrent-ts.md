# CTS（Clear Lab / SUSTech 项目页）

- **类型**：研究项目页（原始摘录）
- **收录日期**：2026-09-22
- **项目页**：<https://clearlab-sustech.github.io/concurrentTS/>
- **论文**：IEEE Robotics and Automation Letters (RA-L), 2024；DOI：<https://doi.org/10.1109/LRA.2024.3457379>；预印本 [arXiv:2405.10830](https://arxiv.org/abs/2405.10830)

## 一句话

**CTS（Concurrent Teacher–Student）** 项目页展示：特权教师与可部署学生在 **同一 PPO 循环** 中并发训练，使四足与点足双足在 uneven terrain 上实现鲁棒敏捷 locomotion；仿真相对两阶段 teacher–student **平均速度跟踪误差最多降约 20%**。

## 为什么值得保留

- 官方 **视频、训练管线图与摘要** 入口；与 [arXiv PDF](https://arxiv.org/pdf/2405.10830) 互补。
- **步骤 2.5 开源核查（2026-09-22）**：项目页 **未列出** 官方 GitHub / 权重下载；仅有演示与补充材料。**工程复现** 见 [Robot-Nav/legbot_lab](https://github.com/Robot-Nav/legbot_lab) 分支 `PPO-CTS-MOE`（MoE 扩展，非 CLEAR Lab 官方仓）。

## 项目页摘录

来源：<https://clearlab-sustech.github.io/concurrentTS/>（2026-09-22 抓取）

- **标题**：*CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion*
- **作者**：Hongxi Wang *、Haoxiang Luo *、Wei Zhang、Hua Chen（* 同等贡献）
- **单位**：
  - Southern University of Science and Technology（南科大 SDIM）
  - Zhejiang University–University of Illinois Urbana-Champaign Institute（ZJUI）
  - LimX Dynamics（逐际动力）
- **摘要口径**：教师与学生 **concurrently** 在 RL 范式下训练；修改版 **PPO** 利用两组策略与环境交互的样本；四足与点足双足室内外实验；相对两阶段 teacher–student 盲 locomotion 速度跟踪误差 **up to 20%** 降低。
- **训练管线（页面图）**：非对称 actor–critic；teacher/student **共享** critic 与 policy；动作由观测 + 特权/本体 encoder 的 latent 决定；特权 encoder 走 policy gradient，本体 encoder 最小化 **reconstruction loss**。

## 开源状态

| 类别 | 状态 | 说明 |
|------|------|------|
| 官方代码 | **项目页未列 GitHub** | 截至 2026-09-22 页内无 Code 链接 |
| 第三方实现 | **已开源** | [Robot-Nav/legbot_lab](https://github.com/Robot-Nav/legbot_lab) `PPO-CTS-MOE`（MoE-CTS + Isaac Lab 部署栈） |

## 对 wiki 的映射

- [`wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md`](../../wiki/entities/paper-cts-concurrent-teacher-student-locomotion.md)
- [`sources/papers/legbot_cts_arxiv_2405_10830.md`](../papers/legbot_cts_arxiv_2405_10830.md)
- [`sources/repos/legbot_lab.md`](../repos/legbot_lab.md)

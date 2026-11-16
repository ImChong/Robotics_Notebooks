# sdpg_visual_rl_arxiv_2605_26478

> 来源归档（ingest）

- **标题：** Efficient On-policy Visual-RL via Stochastic Decoupled Policy Gradient
- **类型：** paper
- **作者：** Haoxiang You, Yilang Liu, Davis Zong, Qian Wang, Teeratham Vitchutripop, Qi Wang, Daniel Rakita, Ian Abraham（Yale / SJTU / Sydney）
- **arXiv：** <https://arxiv.org/abs/2605.26478>
- **项目页：** <https://haoxiangyou.github.io/sdpg-website/>
- **代码：** <https://github.com/HaoxiangYou/SDPG>（**已开源**）
- **入库日期：** 2026-09-06
- **核对范围：** arXiv:2605.26478v1 PDF（26 May 2026）全文通读：正文 §1–§7、附录 A（相关工作）、B（B.1–B.4 推导与证明，只核对结论与式子形式，未逐步复算）、C（C.1–C.5 含 Algorithm 1）、D、E（E.1–E.2，Table 2–6）；图表含 Fig.8 曲线（仅看趋势与横轴范围，未取精确数值）。未读：项目页视频、官方仓库代码（只依赖已有 repo/site 归档）。
- **通读：** 已完成
- **一句话说明：** 随机解耦策略梯度（Stochastic Decoupled Policy Gradient）：用随机扰动 rollout 估计轨迹梯度，混合 batch-rendered 与 physics-only 并行环境，在单卡 RTX 4080 上数小时内端到端训练视觉 MuJoCo 与 egocentric 机器人任务，并展示 Unitree Go2 零样本 sim2real。

## 核心论文摘录（MVP）

### 1) 问题与动机（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2605.26478>
- **核心贡献：** 视觉 RL 比状态 RL 更耗算力与显存：DrQ-v2 / DreamerV3 需大量序贯更新；PPO 类 on-policy 方法需数千并行环境 batch 渲染（如 4096 env × RGB）易 OOM。蒸馏路线（teacher 状态 → student 视觉）快但受信息不对称与分布偏移限制。一阶可微仿真（decoupled PG）高效但梯度不稳、需软接触、仿真器支持差。
- **对 wiki 的映射：**
  - [SDPG 视觉 RL 论文实体](../../wiki/entities/paper-sdpg-visual-rl-stochastic-decoupled.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)

### 2) 随机解耦策略梯度（§3）

- **链接：** <https://haoxiangyou.github.io/sdpg-website/>
- **核心贡献：**
  - 用 **Gaussian 扰动** 平滑估计 \(\nabla_{\mathbf{A}}\mathcal{J}\)，避免全长轨迹反传；结合 **decoupled**（观测 stop-gradient）降低渲染路径显存。
  - **混合环境：** batch-rendered env 评估策略表现；physics-only env 提供扰动 rollout 做策略改进 → **数量级更少** 的 batch-rendered 并行数（论文表：~10 GB vs PPO 视觉 ~48–50 GB）。
  - **工程：** 自适应探索、reward-invariant 归一化稳定更新。
- **对 wiki 的映射：**
  - [Genesis 仿真器](../../wiki/entities/genesis-sim.md)（官方实现基于 Genesis + Hydra）
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)

### 3) Egocentric 任务套件与 sim2real（§4–5）

- **链接：** <https://github.com/HaoxiangYou/SDPG>
- **核心贡献：**
  - 发布 **egocentric** 视觉 RL benchmark：灵巧操作 + 困难 locomotion；RGB/depth、单/多相机 + 本体感知。
  - **Unitree Go2：** RealSense 深度 egocentric 导航崎岖地形/楼梯；仿真 **<2 h** 单 GPU 训练，**零样本** 上真机。
- **对 wiki 的映射：**
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [GR00T Visual Sim2Real](../../wiki/entities/gr00t-visual-sim2real.md)（对照：teacher-student 蒸馏 vs 端到端视觉 on-policy）

## BibTeX（项目页）

```bibtex
@misc{you2026efficientonpolicyvisualrlstochastic,
      title={Efficient On-policy Visual-RL via Stochastic Decoupled Policy Gradient},
      author={Haoxiang You and Yilang Liu and Davis Zong and Qian Wang and Teeratham Vitchutripop and Qi Wang and Daniel Rakita and Ian Abraham},
      year={2026},
      eprint={2605.26478},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.26478},
}
```

## 当前提炼状态

- [x] 摘要与项目页方法图对齐
- [x] 项目页核查：**已开源**（GitHub + 安装/训练 README）
- [x] wiki 页面映射确认

## 通读补充摘录（v1 全文，章节号为论文内编号）

- **Abstract / §1：** 方法为轻量视觉 RL，称单张 RTX 4080 数小时内端到端训多种 visuomotor 策略；「orders of magnitude fewer batch-rendered environments」；在 visual MuJoCo 上「consistently outperforms baseline methods in training time, memory usage, and rewards」（作者自述）。
- **§3.1：** 式 (1) 用 M 个高斯扰动 E^j 的回报差加权平均估计 ∇J_smooth，δ 为探索因子、σ 为回报标准差；Theorem 1（证明见 B.2）：∇J_smooth = (δ/σ)∇_A J_δ。Fig.3：SAPO 梯度范数有大尖峰，SDPG 平稳（1D 玩具例 + 训练梯度范数）。称「avoid backward passes, reducing computation time by half」。
- **§3.1 Prop.1 / 式 (3)(4)：** 把「θ += β (∂A/∂θ)ᵀ d(A)」写成 ½‖A − sg(A + d(A))‖² 的梯度下降；SDPG 损失即 ‖π(O|θ) − sg(A + ∇J_smooth)‖²。
- **§3.2：** REINFORCE 型策略梯度（式 5，基线取均值动作回报）是 SDPG 的 M=1 特例（脚注 6：分母 δ 换成 σ 以数值稳定）。Theorem 2：decoupled PG 等于行为克隆损失（式 7）的负梯度，Fig.5 把方法按「每批 rollout 的轨迹搜索 / BC 迭代数」归类（GPS、NAF、PI-GPS、SHAC、DPG、REINFORCE、A2C、SDPG）。
- **§4：** 演员–评论家，评论家吃低维特权状态、演员吃真机可得观测；H 步 + 价值自举（式 8），TD(λ)（式 9）；N 个 nominal 环境（批量渲染）各配 M 个仅状态 auxiliary 环境（Fig.6），每段 rollout 开始 auxiliary 重置到 nominal 状态；式 (10) 学习探索因子 δ（log δ 参数化）。
- **§5.1 / Fig.8 / Table 1：** Visual MuJoCo（Hopper、Walker、Ant、Humanoid）在 Genesis 中重实现，演员仅见 84×84 三帧第三人称 RGB（E.1.1）；基线 DrQv2、DreamerV3、teacher-student 蒸馏（teacher 用 rl-games 与 RSL-RL 的 PPO）；DrQv2 与 DreamerV3 经并行 / 代码优化（DrQv2 提速约 2 倍，DreamerV3 提速有限）。显存（GB，除 PPO 外均 64 批渲染环境）：Ours 10.2 / 10.3 / 10.3 / 10.5；PPO† 48 / 48 / 49 / 50（†按 4096 环境与状态版超参估计，非实测）；DrQv2 10.6 / 8.2 / 10.5 / 11.6；DreamerV3 10.8 / 10.8 / 10.8 / 10.9；蒸馏 10.6 / 10.6 / 10.3 / 10.7。
- **§5.1 结果文字：** 「achieves the highest rewards, matching state-based performance across all tasks」；与蒸馏训练时间相当、明显快于 DrQv2 / DreamerV3；Fig.8 图注：人形任务最终回报更高。未与 You et al.（NeurIPS 2025，可微仿真版 decoupled PG）直接比较，理由是当时 Genesis 缺刚体可微。Fig.8 横轴（读图，近似）：Hopper 约 1.5 h，Walker 约 1.5 h，Ant 约 3 h，Humanoid 约 20 h。
- **§5.1 蒸馏对照：** Fig.9：人形任务蒸馏学生停在次优并在某些状态失败；Fig.10 G1 hurdle：height-map 型 teacher 难训，端到端 RGB 更好（作者解释）；Fig.11：同一超参下状态输入与视觉输入最终回报相近，视觉需 2–4 倍迭代才收敛。
- **§5.2 / E.1.2 / Table 4：** egocentric 套件共 7 个环境：WalkerHurtle、G1Hurtle、Go2Terrain（RGB/深度，1 相机）、FrankaPickCube（RGB，1 相机）、AlohaInsertion（RGB，3 相机）、AllegroHand、ShadowHand（RGB，1 相机，固定外部相机）。训练曲线在 Fig.13（附录 D.1），正文只给轨迹示意（Fig.7）。
- **§5.3 / D.2：** Go2 用 RealSense 深度，仿真训练 2 小时内，零样本上真机爬楼梯（Fig.12）；附录 Fig.14 另有箱子、大石块、下楼梯的照片（未给成功率等定量指标）。
- **§6 局限（作者自述）：** 总成本仍由物理仿真主导，rollout 每次迭代后丢弃，更接近早期 A2C 而非 PPO 的样本复用；对随机种子敏感，偶发训练不稳定。
- **E.2：** 全部环境 N=64 nominal、M=63 auxiliary，共 4096 个并行物理环境（Table 5）；AlohaInsertion 用 N=48（四个图像编码器显存）；Adam、γ=0.99、λ=0.95；各任务 H、学习率、头宽见 Table 6（附录 E.2）。注意：4096 个物理环境中只有 64 个批渲染。
- **未在论文中找到：** Go2 真机的定量成功率 / 多次重复；Fig.8 的种子数与误差带说明（图中有阴影，正文未交代）；egocentric 套件与基线的对比曲线（只给 SDPG 自身曲线）。

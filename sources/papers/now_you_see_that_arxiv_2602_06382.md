# Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels（arXiv:2602.06382）

> 来源归档（ingest）

- **标题：** Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels
- **类型：** paper / humanoid locomotion / end-to-end vision / depth augmentation / privileged distillation / multi-critic / multi-discriminator / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2602.06382>
- **arXiv HTML：** <https://arxiv.org/html/2602.06382v1>
- **PDF：** <https://arxiv.org/pdf/2602.06382>
- **项目页：** <https://hellod035.github.io/Now_You_See_That/>
- **代码：** <https://github.com/Hellod035/Now_You_See_That>（README + 视频链接；训练代码待发布）
- **会议：** RSS 2026（Accepted，项目页标注）
- **作者：** Wandong Sun*, Yongbo Su*, Leoric Huang*, Alex Zhang, Dwyane Wei, Mu San, Daniel Tian, Ellie Cao, Baoshi Cao, Yang Liu, Finn Yan†, Ethan Xie†, Zongwu Xie†
- **机构：** 哈尔滨工业大学；HONOR Robotics Team
- **入库日期：** 2026-06-11
- **一句话说明：** **两阶段** 人形端到端深度 locomotion：特权 **height scan** 上 **多 critic + 多 discriminator AMP** 学统一 teacher，再以 **vision-aware DAgger 蒸馏**（行为克隆 + 去噪 + KL）把知识迁到 **8 步深度增广管线** 下的 **24×32 立体深度**；在 **RDT-Bench** 与两台人形（Orbbec Gemini 336L / RealSense D435i）上零样本完成极端跑酷与 **30+ 级双向长楼梯**。

## 摘要级要点

- **问题：** 视觉人形 locomotion 受 **sim-to-real 感知噪声**（厘米级落脚任务尤甚）与 **异质地形奖励冲突**（单 value 难覆盖楼梯/沟壑/粗糙地形）双重制约。
- **核心思路：** 端到端框架同时攻 **感知迁移** 与 **统一多地形控制**：
  1. **Realistic depth sensor simulation** — 8 步增广管线模拟立体匹配孔洞、距离相关噪声、Perlin 结构化噪声、光学畸变、标定漂移、像素失效与裁剪延迟。
  2. **Vision-aware behavior distillation** — 特权 height scan teacher → 增广深度 student；**DAgger** 式在线收集 + $\mathcal{L}_{behavior}$ + **去噪一致性** $\mathcal{L}_{denoise}$ + **KL 正则** $\mathcal{L}_{kl}$（$\lambda=0.1$）。
  3. **Multi-critic & multi-discriminator terrain learning** — $K=3$ 地形类（楼梯/平台、沟壑、粗糙地形）各配 **critic head + AMP discriminator**；共享 backbone、分地形 reward shaping 与 motion prior。
- **训练管线：**
  - **Stage 1 特权 RL：** 1.6 m×1.0 m ego-centric height scan（0.05 m 分辨率，$21\times33=693$ 点）+ PPO + 分地形 critic/discriminator + 全面动力学 DR。
  - **Stage 2 视觉蒸馏：** student 输入 **增广立体深度**（30×40 → 裁剪 24×32，量程 0.3–2.0 m）+ 本体；teacher 仍用 clean height scan。
- **深度增广管线（Table II 摘要）：** 启动期随机内参/外参/观测延迟 → 每帧：立体融合 → 随机卷积 → 高斯噪声 → Perlin 噪声 → 尺度随机 → 像素失效 → 深度裁剪 → 空间裁剪。
- **评测基准 RDT-Bench：** 用 **CycleGAN** 把仿真/真机非配对深度对齐，**仅评测期** 注入真实风格噪声；四地形（上/下楼梯、45 cm 沟、40 cm 台）× 1024 env × 100 ep；指标 **SR / 平均功率 P / 功率退化比 PDR**。
- **主要仿真结果（RDT-Bench，Table IV）：** 本文 **98.9%** 平均 SR、**5.8%** PDR；优于 Humanoid Parkour Learning [60]（71.0% / 30.9%）、Single Critic/Disc.（82.0%）、Direct RL（54.0%）、BC Only（86.0%）。
- **增广消融（Table V）：** 去掉 **Stereo Fusion** 降幅最大（98.9→90.4% SR）；Calibration Uncertainties、Depth-Dependent Noise 次之。
- **蒸馏消融（Table VII）：** 去掉 $\mathcal{L}_{denoise}$ → 93.4% SR；去掉 $\mathcal{L}_{kl}$ → 96.1%；仅 BC → 86.0%。
- **真机（Table VIII，15 trial/场景）：** 户外楼梯上 **100%**、下 **86.7%**、平台上下 **100%**、**30+ 级长楼梯 100%**、宽沟 **100%**；总 **88/90 = 97.8%**；机载 **50 Hz** 无微调。
- **跨平台：** 主平台 **Orbbec Gemini 336L** 立体深度；同管线迁移至 **Unitree G1 + Intel RealSense D435i**（附录 E 初步结果）。
- **项目页实机场景：** Wild Parkour、上下楼梯、垫脚石、室内跑酷、平衡恢复等。

## 核心摘录（面向 wiki 编译）

### 两阶段训练（Sec. III / Fig. 4）

| 阶段 | 观测 | 机制 | 输出 |
|------|------|------|------|
| **Privileged RL** | Height scan 693 维 + 本体 | $K=3$ critic + $K=3$ discriminator AMP；分地形 reward | Teacher policy $\mu_{priv}$ |
| **Vision-aware Distillation** | 增广深度 24×32 + 本体 | DAgger rollout + $\mathcal{L}_{total}$ | Deploy policy $\mu_{deploy}$ |

### 深度增广八步（Sec. III-A / Table II）

1. **Stereo fusion** — 左右视差一致性阈值 $\tau\in[0.05,0.20]$，失败像素置零（孔洞）。
2. **Random convolution** — $3\times3$ 核模拟光学像差。
3. **Gaussian noise** — $\sigma(d)=|c_0+c_1 d+c_2 d^2|$ 距离相关。
4. **Perlin noise** — 5 octave 结构化干扰。
5. **Scale randomization** — $s\sim\mathcal{U}(0.90,1.10)$ 标定漂移。
6. **Pixel failures** — $p_{zero}=p_{max}=0.001$ 死/饱和像素。
7. **Depth clipping** — $[0.3, 2.0]$ m。
8. **Spatial cropping** — $30\times40 \to 24\times32$。

### 多 critic / 多 discriminator（Sec. III-B）

- 地形类 $k\in\{1,2,3\}$：楼梯与平台（含上下）、沟壑、粗糙地形。
- $V(s_t)=V_{k(s_t)}(s_t)$，$D(s_t)=D_{k(s_t)}(s_t)$；critic 共享 backbone、独立 value head。
- 每类独立 AMP motion dataset。

### 与相邻路线对比（Table I / 索引级）

| 维度 | Now You See That | Humanoid Parkour [60] | SSR | PHP | Extreme Parkour |
|------|------------------|----------------------|-----|-----|-----------------|
| 表示 | **端到端深度** | 端到端深度 | 端到端深度 | 端到端深度 | 端到端深度 |
| 噪声建模 | **全面 8 步增广** | Moderate DR | Moderate | Moderate | Moderate |
| 多地形统一 | **多 critic + 多 disc** | 单策略 | 分地形 AMP | 多技能蒸馏 | 单策略 |
| 训练阶段 | **特权 height → 深度蒸馏** | 直接/标准 DR | **单阶段 PPO** | MM 参考 + 专家 + DAgger | scandots → 深度 DAgger |
| 精细楼梯 | **✓ 双向长程** | ✗ | ✓ | ✓ | ✗ |
| 极端跑酷 | **✓** | ✓ | ✗ | ✓ | ✓（四足） |
| 长程无漂移 | **✓ 30+ 级楼梯** | ✓ | **✓ 1.3 km 户外** | ✓ | ✗ |

### 局限（Sec. V）

- **下楼弱于上楼**（86.7% vs 100%）：重力放大误差 + 目标落脚被台阶边缘遮挡（立体匹配伪影高发区）。
- 未来方向：近场更高分辨率深度、预测性落脚。

## 对 wiki 的映射

- 沉淀实体页：[Now You See That 端到端视觉人形 locomotion（arXiv:2602.06382）](../../wiki/entities/paper-now-you-see-that-humanoid-vision-locomotion.md)
- 交叉更新：[humanoid-locomotion.md](../../wiki/tasks/humanoid-locomotion.md)、[stair-obstacle-perceptive-locomotion.md](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[privileged-training.md](../../wiki/concepts/privileged-training.md)、[sim2real.md](../../wiki/concepts/sim2real.md)、[dagger.md](../../wiki/methods/dagger.md)
- 姊妹对照：[SSR](../../wiki/entities/paper-ssr-humanoid-open-world-traversal.md)、[PHP](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[Humanoid Parkour Learning（Zhuang et al.）](../../wiki/entities/paper-notebook-humanoid-parkour-learning.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2602.06382>
- 项目页：<https://hellod035.github.io/Now_You_See_That/>
- 代码仓：<https://github.com/Hellod035/Now_You_See_That>
- 深读笔记（姊妹仓）：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Now_You_See_That_Learning_End-to-End_Humanoid_Locomotion_from_Raw_Pixels/Now_You_See_That_Learning_End-to-End_Humanoid_Locomotion_from_Raw_Pixels.html>

# VMP: Versatile Motion Priors for Robustly Tracking Motion on Physical Characters

> 来源归档（ingest · Disney Research PDF）

- **标题：** VMP: Versatile Motion Priors for Robustly Tracking Motion on Physical Characters
- **类型：** paper / physics-based-animation / motion-tracking / motion-prior / character-control / sim2real
- **机构：** ETH Zürich、Disney Research（瑞士）
- **作者：** Agon Serifi, Ruben Grandia, Espen Knoop, Markus Gross, Moritz Bächer
- **发表：** SCA 2024（ACM SIGGRAPH/Eurographics Symposium on Computer Animation，Montreal，2024-08）
- **DOI：** <https://doi.org/10.1111/cgf.15175>
- **PDF：** <https://la.disneyresearch.com/wp-content/uploads/VMP_paper.pdf>
- **项目页：** <https://cgl.ethz.ch/publications/papers/paperSer24a.php>
- **Paper Notebooks 分类：** 04_Loco-Manipulation_and_WBC（待深读清单锚点）
- **入库日期：** 2026-06-11（PNB 锚点）；2026-06-29（PDF 深读 ingest）
- **一句话说明：** 两阶段「β-VAE 运动潜空间 + 条件 PPO 跟踪」：从未过滤大规模动捕学 versatile motion prior，用全身运动学参考 + 时变 latent 驱动单一策略，在仿真与 LIME 双足真机上实现高精度、可艺术家导向的物理角色跟踪。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| PDF | <https://la.disneyresearch.com/wp-content/uploads/VMP_paper.pdf> | 本次 ingest 主来源 |
| DOI | <https://doi.org/10.1111/cgf.15175> | Computer Graphics Forum / SCA 2024 |
| ETH 页 | <https://cgl.ethz.ch/publications/papers/paperSer24a.php> | 摘要与下载 |
| 对照 | CALM [TKG*23]、ASE [PGH*22]、DeepMimic [PALVdP18] | 端到端 latent vs 显式跟踪 vs 对抗先验 |
| 硬件线 | BDX / LIME 角色机器人 [GKH*24] | 同一 Disney 双足角色研究线 |

## 摘要级要点

- **问题：** 从未结构化动捕学单一控制策略，使其对**多样且未见**运动保持高精度跟踪，并可部署到真机，仍很困难。
- **两阶段解耦：** Stage I 用 **β-VAE** 在短运动窗口上自监督学 kinematic latent；Stage II 用 **PPO** 训练条件策略 $\pi(a_t|s_t, m_t, z_t)$，$m_t$ 为当前参考帧、$z_t$ 为以该帧为中心的窗口 latent——分离自监督表征与显式 imitation reward，避免对抗训练的 mode collapse。
- **接口：** 用户/艺术家提供 **全身运动学参考序列**；encoder 映射为时变 latent 流，policy 输出力矩/PD 目标，实现 physics-informed 全身控制。
- **规模：** **11 h** 未过滤数据（CMU 8.5 h + Mixamo 2 h + Reallusion 0.5 h）；36-DoF 人形 + **LIME** 20-DoF 双足（0.84 m，16.2 kg）真机验证。
- **相对 CALM：** latent 可分性更高（LDA **0.854 vs 0.687**）；完整管线 RTX 4090 **<3 天** vs CALM A100 **~2 周**；未见舞蹈/格斗跟踪误差约 **5°** 关节 MAE（全数据训练）。

## 核心摘录（面向 wiki 编译）

### 1) Stage I：Versatile Motion Prior（β-VAE）

- **运动窗口：** $M_t=\{m_{t-W},\ldots,m_{t+W}\}$，$W=30$（约 1 s）；每帧 $m_t=\{h_t,\theta_t,v_t,q_t,\dot q_t,p_t\}$（根高、6D 朝向、速度、关节角/速、手足相对根位置）。
- **归一化：** 以中心帧根朝向为局部 heading frame；除朝向外其余量按数据集均值/方差归一化。
- **网络：** 1D Conv + ConvResNet blocks；$d_z=64$；β-VAE，$\beta=0.002$，KL 周期调度；RTX 4090 训练 **~10 h**。
- **设计要点：** **每帧一个 latent**（非整段单码），便于细粒度控制、即时响应参考突变与空间组合。

### 2) Stage II：条件 PPO 跟踪策略

- **条件：** $c_t=(m_t, z_t)$；策略为 3 层 512 MLP + 对角高斯；**Isaac Gym** 8192 并行环境，PPO **~48 h**。
- **奖励：** $r_t=r^{\text{track}}_t+r^{\text{alive}}_t+r^{\text{smooth}}_t$——根/关节/末端跟踪 MSE + 存活 + 动作一阶/二阶平滑与力矩惩罚。
- **终止：** 末端偏差超阈持续 $f$ 帧则 early terminate（非仅足接触终止），允许全身着地。
- **鲁棒性：** 质量/摩擦/推扰域随机化；**执行器模型**（PD 电机 + Coulomb 摩擦 + 速度相关力矩限）；机器人额外关节标定噪声 $\epsilon_q$。

### 3) 消融与对比（人形，Reallusion 子集 + 未见集）

| 条件输入 | Idle | Walk | Attack | Dance | Unseen（关节 MAE °） |
|----------|------|------|--------|-------|----------------------|
| M（仅当前帧） | 7.52 | 8.63 | 13.11 | 12.79 | 13.29 |
| L（仅 latent） | 6.10 | 7.53 | 10.70 | 10.45 | 10.92 |
| **LM（本文）** | **4.31** | **4.15** | **7.08** | **5.80** | **7.83** |

- 全数据训练后未见动作 MAE **~5°**；对不可行参考（空中爬楼梯）尽量跟踪并保持平衡。
- 相对 CALM：训练更快、latent 信息更丰富；LM 耦合使策略对参考更「跟手」。

### 4) 艺术家导向与真机

- **空间组合：** 不同 clip 的臂/身可拼接；**运动编辑：** 任意排序片段后精调关键帧/风格化。
- **LIME 真机：** 无踝 roll 时策略自适应用脚尖触地维持平衡；动态动作在物理执行器极限内仍高保真跟踪。
- **局限：** 长规划视野特技（后空翻等）需更强记忆结构；生成式遍历 latent 空间尚未展开。

## 对 wiki 的映射

- [paper-notebook-vmp](../../wiki/entities/paper-notebook-vmp.md)
- [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)
- [Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md)
- [DeepMimic（方法）](../../wiki/methods/deepmimic.md)
- [人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)
- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)

## 参考来源（原始）

- [VMP PDF](https://la.disneyresearch.com/wp-content/uploads/VMP_paper.pdf) — 本次 ingest 主来源
- [Humanoid Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/progress.json) — 待深读清单锚点

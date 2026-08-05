# RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking（arXiv:2509.20717）

> 来源归档（ingest）

- **标题：** RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking
- **类型：** paper / humanoid motion tracking / residual-action RL / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2509.20717>
- **arXiv HTML：** <https://arxiv.org/html/2509.20717v2>
- **PDF：** <https://arxiv.org/pdf/2509.20717>
- **作者：** Zhenguo Sun*、Yibo Peng*（* equal）、Yuan Meng、Xukun Li、Yu Sun、Haojun Jiang、Bo-Sheng Huang、Zhenshan Bing†、Xinlong Wang†、Alois Knoll（† corresponding）
- **机构：** 慕尼黑工业大学（Technical University of Munich）；北京智源人工智能研究院（BAAI）；XYZ Embodied AI；清华大学（Tsinghua University）；南京大学（Nanjing University）
- **发表信息：** 2025-09-25 arXiv v1；2026-08-03 arXiv v2（accepted by IEEE Robotics and Automation Letters / R-AL）；稿件 received 2026-04-05，revised 2026-07-03，accepted 2026-08-01
- **入库日期：** 2026-08-05
- **平台：** Unitree G1（主结果）；H1 / H1-2（跨平台 sim-to-sim + 定性真机片段）
- **数据：** 8 段完整 LAFAN1 舞蹈参考（retarget 到 G1/H1/H1-2）
- **开源状态（步骤 2.5，截至 2026-08-05）：** **确认未开源** — arXiv abs / HTML / PDF **无项目页与 GitHub 链接**；附录称 “released robot configuration files” 提供平台 PD / 限位等，但 **未给出可访问 URL**；GitHub 检索无官方 `RobotDancing` 仓（无关同名玩具仓除外）
- **一句话说明：** 长时程高动态人形追踪脆在「参考–机器人动力学失配」累积；RobotDancing 用**参考条件残差关节目标** $q^{\mathrm{tar}}=q^{\mathrm{ref}}+a$（选择性仅髋/膝 pitch）+ 分布均衡与失败优先采样，单阶段 PPO 配方在 G1 上零样本跑通 8 段 LAFAN1 舞蹈（21/24 真机成功）。

## 摘要级要点

- **痛点：** Retarget 参考往往运动学可行但动力学不一致；绝对关节指令跟踪在长序列高能动作上误差累积直至失稳。
- **主张：** **Track by correcting** — 策略只学参考上的有界残差，把容量用在补偿驱动、接触、延迟与 retarget 误差，而不是重合成整段动作。
- **配方属性：** **per-sequence**（每条参考训一个策略）+ **跨动作/跨平台复用**同一套观测、奖励、超参与部署流程；不是通用多技能 tracker。
- **四贡献：**（1）可复现单阶段长时程追踪配方；（2）参考条件选择性残差动作；（3）分布感知均衡 + 失败感知优先采样；（4）八动作 × 三平台评估。

## 核心摘录（面向 wiki 编译）

### 1) 残差动作与选择性掩码（§III-B / III-D）

| 符号 | 含义 |
|------|------|
| $O_t=[P_t;G_{t+1}]$ | actor 观测：本体感觉 + 下一帧参考 |
| $a_t$ | 残差关节位置目标（G1/H1-2/H1 维数 23/21/19） |
| $q^{\mathrm{tar}}_t=q^{\mathrm{ref}}_{t+1}+a_t$ | 残差叠加后进 PD |
| $\mathbf{m}\odot a_t$ | 选择性掩码：仅双侧髋/膝 pitch（G1 索引 `[0,3,6,9]`；踝 pitch 与躯干/臂默认不残差） |

- 非对称 actor–critic：critic 额外吃特权基座速度、参考 link 位姿与随机化物理参数。
- 奖励：$r=r^{\mathrm{track}}-s_{\mathrm{pen}}(t)\,r^{\mathrm{reg}}$；tracking 用 DeepMimic 式高斯核；每价值头独立 GAE 后求和进 PPO。

### 2) 长尾采样（§III-C）

- 参考切成 **1 s** 段；**Distribution-Aware Balancing** 在髋/膝 pitch 子空间做直方图均衡先验 $p^{\mathrm{bal}}$。
- **Failure-Aware Prioritization**：段级 EMA 失败率 → $p^{\mathrm{fail}}$。
- **Bounded Mixture**：$\lambda_u=0.3$，失败/分布权重从 $(0.4,0.2)$ ramp 到 $(0.55,0.05)$（48K warmup + 48K ramp）；概率帽防塌缩。
- RSI：从采样参考态起步并加小扰动。

### 3) 关键评测数字（§IV）

| 设置 | 指标 | 结果 |
|------|------|------|
| 八动作残差对比（Table III） | SELECTIVE vs NONE：$E_{g\text{-}mpbpe}$ / $E_{mpbpe}$ / $E_{mpjpe}$ | **−15.7% / −18.2% / −20.5%**；多数指标亦优于 ALL-DOF 残差 |
| dance1_subject2 基线（Table V，3 seeds） | Fixed completion / Start-zero | RobotDancing **97.7% / 131.5 s** vs ASAP-style 3.3% / 5.7 s、KungfuBot-style 4.3% / 13.5 s（作者重实现，非原文数字） |
| dance2_subject4 掩码消融（Table VI） | Comp. / Surv. | SELECTIVE **38.6% / 45.0 s** 最高鲁棒；扩大残差权威可降局部 MPJPE 但常损完成率 |
| 采样消融（Table VII） | 30 s threshold success | Combined **26.1%** > Failure-only 22.0% > Distribution-only 18.6% > Uniform RSI 10.0% |
| G1 真机（Table IV） | 8 序列 × 3 次 | **21/24（87.5%）** 全程成功；MuJoCo 验证后 TorchScript @ 50 Hz Orin NX，无测试期滤波/重缩放 |
| 跨平台 sim（Table VIII，dance1_subject2） | Start-zero | G1/H1 **131.5 s**；H1-2 **~34 s**（motion-far 阈值，非跌倒） |

### 4) 局限（§V）

1. 每序列一策略，非通用 tracker。
2. 残差掩码仍是形态启发式。
3. H1/H1-2 真机仅为定性片段。
4. 缺同步力矩/接触/滑移/基座姿态遥测，硬件失败诊断受限。

## 对 Wiki 的映射

| 主题 | 关系 |
|------|------|
| [RobotDancing（论文实体）](../../wiki/entities/paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md) | **主沉淀页**（原 Paper Notebooks stub，本次升格） |
| [深读笔记锚点](./humanoid_pnb_robotdancing.md) | 姊妹仓库笔记溯源 |
| [Residual Policy Learning](../../wiki/methods/residual-policy-learning.md) | 「参考轨迹作 base + 残差」谱系边界案例 |
| [ASAP](../../wiki/entities/paper-notebook-asap-aligning-simulation-and-real-world-physics.md) / [KungfuBot](../../wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md) | 论文 Table V 同协议重实现对照 |
| [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) / [Sim2Real](../../wiki/concepts/sim2real.md) | 参考–动力学失配与零样本部署语境 |
| [Unitree G1](../../wiki/entities/unitree-g1.md) | 主真机平台 |
| [Physics-Based Animation 分类](../../wiki/overview/paper-notebook-category-13-physics-based-animation.md) | Paper Notebooks 分类父节点 |

## BibTeX（arXiv）

```bibtex
@misc{sun2026robotdancing,
  title={RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking},
  author={Sun, Zhenguo and Peng, Yibo and Meng, Yuan and Li, Xukun and Sun, Yu and Jiang, Haojun and Huang, Bo-Sheng and Bing, Zhenshan and Wang, Xinlong and Knoll, Alois},
  year={2026},
  eprint={2509.20717},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  note={Accepted by IEEE Robotics and Automation Letters}
}
```

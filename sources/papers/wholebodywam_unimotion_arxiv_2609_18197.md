# WholeBodyWAM · UniMotion-4K（arXiv:2609.18197）

> 来源归档（paper）

- **标题：** WholeBodyWAM: Learning Whole-Body World Action Models with Scalable Motion Priors
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.18197>
- **PDF：** <https://arxiv.org/pdf/2609.18197>
- **项目页：** <https://zbzyjya.github.io/WholeBodyWAM/>
- **机构：** 南开大学（Nankai University）；北京人形机器人创新中心（Beijing Innovation Center of Humanoid Robotics）；北京理工大学（BIT）；清华大学（Tsinghua University）
- **作者：** Bowei Zhang, Qiyao Zhang, Shuanghao Bai, Xinhua Wang, Meng Li, Yilei Wang, Leiwang Zhang, Jian Tang, Lu Zhou, Lei Sun, Zhengping Che
- **入库日期：** 2026-09-17
- **一句话说明：** 从 4.1K+ 小时异构全身运动预训练可迁移 motion prior，再经 Video–Motion–Action 两阶段 WAM 接地到天工 3.0 真机 loco-manipulation。

## 同名区分

- 另一篇 **WholeBodyWAM**（arXiv:[2609.16644](https://arxiv.org/abs/2609.16644)，CUHK/HKU/PKU/Φ，WBC-grounded coordination）见 [`wholebodywam_arxiv_2609_16644.md`](wholebodywam_arxiv_2609_16644.md) 与 [`paper-wholebodywam`](../../wiki/entities/paper-wholebodywam.md)。

## 开源状态

- **待发布**（步骤 2.5 核查，2026-09-17）：项目页 Code 按钮标注 **Coming Soon**，无 GitHub / Hugging Face 链接。

## 核心摘录

1. **问题：** 目标机器人全身轨迹昂贵难扩；人类/人形异构 motion 丰富但不能直接当 embodiment-specific action 监督。
2. **UniMotion-4K：** 11 源异构数据 → 约 **1.19M** 序列、**444.4M** 帧、**4.1K+** 小时；统一 **63D** root-free 表示（21 关节局部 axis-angle）。
3. **Stage I：** 30-block **Motion Expert**，flow matching；16 帧历史 + 语言 → 预测未来 32 帧；**无需**目标机器人 action。
4. **Stage II：** **Video / Motion / Action Experts** 层间 **MoT** 联合注意力；非对称 mask——Motion/Action 可读当前视觉，**不可读未来视觉 latent**（防泄漏）。
5. **部署：**  onboard 图像 + 16 motion 状态 + 语言 → 联合预测 future motion + action chunk；**不生成/解码未来视频**；腿高阶命令走 RL WBC。
6. **真机（天工 3.0，6 任务）：** 平均归一化任务分 **72.2%**（GR00T N1.7 **60.8%**）；去 Stage-I **59.1%**；阻断 Motion→Action **46.3%**。
7. **Scaling：** motion pretrain 0→4K+ h，MPJRE **1.127°→0.817°**（**27.5%↓**）；同 setup 下游分 **57.1%→67.6%**；50% 示范时 4K+ 模型 **46.9%** > FastWAM 全量 **42.9%**。
8. **效率：** A100 端到端 **363 ms**（DreamZero **19.5×** 慢）。

**对 wiki 的映射**

- [paper-wholebodywam-unimotion-4k](../../wiki/entities/paper-wholebodywam-unimotion-4k.md)
- [paper-wholebodywam](../../wiki/entities/paper-wholebodywam.md)（同名异文对照）

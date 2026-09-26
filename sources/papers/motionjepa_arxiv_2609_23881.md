# MotionJEPA: Preventing Temporal Feature Collapse by Capturing Visual Changes in Latent Space（arXiv:2609.23881）

> 来源归档

- **arXiv：** <https://arxiv.org/abs/2609.23881>
- **PDF：** <https://arxiv.org/pdf/2609.23881>
- **项目页：** <https://mkarmann.github.io/motion-jepa-project-page/>
- **代码：** <https://github.com/mkarmann/motion-jepa>
- **开源状态：** **已开源**（主仓：Pong/Dino/Golf 离线 probing 与训练；下游 CEM 规划在 `planning/` 子模块，需 `git clone --recurse-submodules`）
- **入库日期：** 2026-09-26
- **机构（项目页）：** 牛津大学；vivo Tech Research GmbH / vivo 蓝图影像实验室；比勒费尔德大学；Slater Labs；冷泉港实验室；布朗大学；AMI Labs
- **一句话说明：** 在标准 JEPA 上加入 DISReg（差分图像 + 单帧 embedding 正则），平衡静态/动态特征、缓解时间特征坍塌；三合成游戏 latent probing 全面 NMSE≤0.1；LeWM 四任务 CEM 在静态背景干扰下 25/50 步规划均值 **81.8% / 70.3%**（相对 LeWM **20.4% / 11.8%**）。

## 核心摘录（对 wiki 编译）

1. **问题：** 无重建 JEPA 预训练偏向 **slow features**，动态信息被抑制；逆动力学可防坍塌但依赖 **动作标签**，对无标注一般动力学激励不足。
   - **映射：** [paper-motionjepa](../../wiki/entities/paper-motionjepa.md)「为什么重要」；对照 [LeWM](../../wiki/entities/paper-lewm.md)、[LeJEPA](../../wiki/entities/paper-lejepa.md)。

2. **DISReg：** 逆动力学风格模块预测 **时间差分图像 embedding** \(\hat d_t\)，**无像素重建**；静态项 SIGReg(\(z\)) 塑形单帧分布，动态项 MSE(\(d_t,\hat d_t\)) 只要求变化信息存在、不约束 \(z\) 形状。总损失 = 下一 embedding MSE + DISReg（默认 \(\lambda_z{=}0.25\), \(\lambda_{\mathrm{pred}}{=}0.5\), \(\lambda_d{=}2\)）。
   - **映射：** 实体页「核心原理」「流程总览」。

3. **表征：** Latent probing（Pong/Dino/Golf）显示 MotionJEPA 比 LeWM* / SMWM* 与各 LeWM 变体更 **完整**；Golf 单特征扫掠 PCA 轨迹 **低曲率**（相对 LeWM* 塌缩、LeWM-Flat* 折叠）。
   - **映射：** 实体页「实验与评测 · 离线 probing」。

4. **规划：** 四 LeWM 控制任务（Cube / PushT / Reach / 2Room）背景换木纹静态干扰，CEM；25 步均值 **81.8%**（LeWM 20.4%，IDM 77.2%）；50 步 **70.3%**（LeWM 11.8%，IDM 59.3%）；相对 IDM 均值 **+4.6 / +11.0 pp**。
   - **映射：** 实体页「规划 under distractors」；方法页 [generative-world-models](../../wiki/methods/generative-world-models.md)。

5. **实现：** 架构与 SIGReg 取自 [LeWM/le-wm](https://github.com/lucas-maes/le-wm)；`uv sync` + `train.py` / `train_probes.py` / `evaluate.py`；基线含 `lewm`、`smwm` 及 LeWM 变体。
   - **映射：** [sources/repos/motion-jepa.md](../repos/motion-jepa.md)；实体页「源码运行时序图」。

**对 wiki 的映射：** [paper-motionjepa](../../wiki/entities/paper-motionjepa.md)

# Paper: SMP (2025/2026)
- **Title**: SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control
- **Authors**: Yuxuan Mu, Ziyu Zhang, Yi Shi, ..., Xue Bin Peng
- **Venue**: arXiv:2512.03028 (v3, 2026)
- **Link**: https://arxiv.org/abs/2512.03028

## 核心贡献
1. **可复用运动先验**：使用冻结的、预训练的扩散模型作为奖励函数，训练阶段无需访问原始动作数据集。
2. **SDS (Score Distillation Sampling)**：将视觉领域的得分蒸馏采样适配到动作控制，引导 RL 策略向自然行为收敛。
3. **ESM (Ensemble Score-Matching)**：通过聚合多个噪声水平的评估结果，显著降低奖励方差，提升训练稳定性。
4. **GSI (Generative State Initialization)**：利用扩散模型生成训练初始状态，彻底摆脱对轨迹数据集的依赖。
5. **风格组合 (Style Composition)**：在无需新数据的情况下，通过混合不同的先验实现风格融合（如“飞机手”+“高抬腿”）。
6. **Unitree G1 部署**：在附录中详细展示了该算法在 Unitree G1 人形机器人上的真机迁移效果。

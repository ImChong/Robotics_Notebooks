# ACG-WAM（arXiv:2610.06965）

> 来源归档（paper；2026-10-08 核对）

- **标题：** ACG-WAM: World-Action Modeling via Action-Conditioned Geometric Latent Prediction
- **论文：** <https://arxiv.org/abs/2610.06965>
- **HTML：** <https://arxiv.org/html/2610.06965v1>
- **作者：** Jiangtao Liu、Zishang Xiang、Yage He、Lingguo Cui、Baihai Zhang、Runqi Chai、Senchun Chai
- **机构：** 北京理工大学自动化学院；逐际动力（LimX Dynamics）
- **项目页：** <https://RoboOpus.github.io/ACG-WAM/>
- **代码：** <https://github.com/RoboOpus/ACG-WAM>
- **权重：** <https://huggingface.co/RoboOpus/ACG-WAM>
- **对 wiki 的映射：** [ACG-WAM](../../wiki/entities/paper-acg-wam-geometric-latent-prediction.md)

## 核心摘录

1. **问题：** 视频预测与动作损失没有直接要求策略表征捕捉“动作带来什么几何变化”；temporal attention 后的当前帧 token 还可能混入未来信息。
2. **方法：** ACG-JEPA 用当前观测特征、示教动作前缀和预测时域预测 VGGT 对当前/未来图像对编码得到的未来几何特征；头部、左腕和右腕三视角提供监督。
3. **训练位置：** 几何损失回传到 temporal mixing 之前的共享 patch embedding；和 Motus 视频与动作目标共同训练。
4. **部署：** VGGT teacher、cache、adapter 和 predictor 在推理时移除，已训练的共享视觉表征与 Motus backbone 保留。
5. **实验：** RoboTwin 2.0 的 Clean / Randomized / Mean 成功率为 93.46 / 92.68 / 93.07%；TRON2 + WUJI hands 三项真机任务平均 SR 85.00%、PCS 91.67%。
6. **开放边界：** 仓库为 Apache-2.0，权重可从 HF 下载；RoboTwin 示教数据、teacher cache 和上游基础模型需另行获取。


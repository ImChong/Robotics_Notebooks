# Diffusion-Based Impedance Learning｜官方项目主页

- **类型：** site（论文项目）
- **项目：** https://strokeairobotics.github.io/Diffusion-Based-Impedance-Learning_T-RO/
- **论文：** https://arxiv.org/html/2509.19696v4
- **代码：** https://github.com/StrokeAIRobotics/Diffusion-Based-Impedance-Learning_T-RO
- **开放状态：** 已开源 Python 扩散训练/推理与 C++ 控制客户端、Vision Pro 采集接口、部分演示数据和 CAD；项目主页提供采集、parkour、三种 peg、消融的视频入口。
- **核对范围：** 已核对作者官方主页正文、实验视频说明、固定增益与扩散自适应的对比、源码链接。视频入口存在，不代表本次播放逐帧复核或独立复现。
- **通读：** 已完成
- **wiki：** [论文唯一详情](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md)

## 项目资料及结果

- 示教：Apple Vision Pro 的 parkour 与康复辅助上肢动作；在真机任务中以 kinesthetic teaching 录入名义 ZFT。
- Parkour：固定增益在障碍中停止，模型自适应阻抗可完成障碍路径。
- Peg-in-hole：圆柱、方形、星形各 30 次自适应实验均成功；固定高刚度分别 30/30、4/30、0/30。
- 消融：直接从名义 ZFT 计算方向性调节无法取代 sZFT 重建；“模型 + 能量估计 + 方向调节”共同构成完整方法。

## 对 wiki 的映射

- [论文节点：方法、实验与工程复现](../../wiki/entities/paper-diffusion-based-impedance-learning-2509-19696.md)
- [官方仓库归档](../repos/diffusion-based-impedance-learning-tro.md)
- [原始论文归档](../papers/diffusion-based-impedance-learning-2509-19696.md)

# 单卡分钟级人形运动跟踪器 BlinkMimic（2611.16422）

> 论文一手原文归档

- **英文题目：** BlinkMimic: Training a Humanoid Motion Tracker in Minutes on a Single GPU
- **arXiv：** https://arxiv.org/abs/2611.16422
- **HTML 全文：** https://arxiv.org/html/2611.16422v1
- **PDF：** https://arxiv.org/pdf/2611.16422v1
- **机构：** 独立作者 Fuyi Chen（arXiv v1 未注明机构）
- **官方代码：** https://github.com/fuyichen0725/blinkmimic
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 1–5 方法/所有消融、6 局限、附录配置，以及 GitHub README/scripts/train.sh、scripts/presets.sh、tools/eval_batch.py；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。另阅读官方 GitHub README 与实际入口脚本。
- **通读：** 已完成
- **一句话说明：** 基于 mjlab/MuJoCo Warp 训练可微世界模型，策略围绕参考姿态输出残差，通过 imagined rollouts 反向传播；Muon 优化、进度学习率、半隐式世界模型、左右镜像、部署条件错误位置重置、CUDA Graph 整轮训练降低迭代与尾部耗时。

## 原文摘录（提炼）
1. **问题与动机：** 运动跟踪训练耗时不仅由前期拟合决定，最后失败 clip 的清除速度与单迭代 host 开销同样关键。
2. **方法：** 基于 mjlab/MuJoCo Warp 训练可微世界模型，策略围绕参考姿态输出残差，通过 imagined rollouts 反向传播；Muon 优化、进度学习率、半隐式世界模型、左右镜像、部署条件错误位置重置、CUDA Graph 整轮训练降低迭代与尾部耗时。
3. **实验：** RTX 4090 单卡：40 段 LAFAN1 共 2.45 小时动作，均值 2.94 min 首次达到 return 0.82；严格满足连续 3 次检查点均无失败 clip 的训练墙钟 3.38 min（3 seeds）。同测 InstantTrack v1.2 严格目标为 7.36 min。未测实体 G1 在线训练。
4. **边界与局限：** 仅 G1/LAFAN1 与单张 RTX 4090；共享 GPU 时异步 eprobe 可能出现非法显存访问。ONNX 测试工具不等于已有真机部署实现。

## 开源状态
官方仓库公开 scripts/train.sh、scripts/presets.sh、src/blinkmimic/train.py、tools/eval_batch.py、tools/export_onnx.py；只核对代码接口，不宣称已在本环境 GPU 重训。

## 对 wiki 的映射
- [BlinkMimic](../../wiki/entities/paper-blinkmimic.md)

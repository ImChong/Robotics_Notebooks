# Adapting Monocular Depth Estimation in Novel Scenes through Disparity Refinement and Robot Ego-Motion

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** Adapting Monocular Depth Estimation in Novel Scenes through Disparity Refinement and Robot Ego-Motion
- **中文：** EM-ViT：机器人运动辅助的单目深度在线尺度适配
- **arXiv：** 2611.14958
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已阅读 arXiv HTML 的摘要、背景、方法、实验、结论、局限及可用附录；公开源码仅核查官方 README 与实际入口，未独立运行。
- **通读：** 已完成
- **论文：** https://arxiv.org/abs/2611.14958
- **HTML：** https://arxiv.org/html/2611.14958
- **PDF：** https://arxiv.org/pdf/2611.14958
- **官方项目页：** https://emvit-mde.github.io/

## 论文机制

读取基础模型的仿射不变逆深度；dRefNet 细化视差图、ViT 预测有效基线尺度；借助稀疏 LiDAR、光度重建和梯度匹配，并可选 odomNet 优化历史相机位姿。

## 实验与证据边界

论文在 KITTI 等室内外公开序列评估深度精度与适应速度；结果针对有稀疏测距/运动信息支持的在线校准，不是完全无标定的单目绝对尺度恢复。

## 复现与开放状态

项目页 Code 链接指向 github.com/YOUR REPO HERE 占位地址，截至 2026-11-14 不能判定代码开放。

## 局限

动态物体、光度重建退化和稀疏 LiDAR 噪声会影响自监督；项目页 Code 链接在核查时仍指向占位仓库，不能声称已发布运行代码。

## 对 wiki 的映射

- [EM-ViT：机器人运动辅助的单目深度在线尺度适配](../../wiki/entities/paper-emvit-online-metric-depth.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2611.14958)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)

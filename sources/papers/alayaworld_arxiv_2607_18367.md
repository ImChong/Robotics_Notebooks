# AlayaWorld — 压缩时序记忆与显式几何缓存的交互世界

- **类型：** paper / 原论文 HTML 与官方源码核对
- **论文：** <https://arxiv.org/abs/2607.18367>
- **代码：** <https://github.com/AlayaLab/AlayaWorld>
- **版本：** 2026-07 全文报告；2026-08 v1.1
- **核对范围：** arXiv HTML 的引言/方法/数据/实验/局限与主要实现附录，结合官方 README 和运行入口核对；未复现实验与 GPU benchmark。
- **通读：** 已完成

## 问题与方法

在双向视频基础模型预训练后，转入因果视频块生成：条件前缀由固定 sink、压缩历史特征、按当前相机从空间记忆重投影的条件和最近帧组成；用户可逐块切换提示。训练通过模型自身 rollout 的退化样本缓解积累漂移，后训练用 DMD + Self-Forcing++ + 一致性蒸馏把约 30 步缩到 4 步。仓库 v1.1 额外提供 DA3 与 ViGeo 两套空间记忆实现，需对齐所用 checkpoint。

## 输入、输出与复现路径

- **输入与输出：** 首帧 `*_image.*`、相机位姿 `*_camera.pt`、文本 `*_prompt.txt`（可选 `*_skill.txt`）；输出按多 chunk 自回归生成的视频。
- **实际入口：** inference/run.py，alaya/inference/pipeline.py，README Quick Start 和 docs/vigeo/README.md
- **官方复现：** `pip install -r requirements.txt`、安装 Depth-Anything-3 依赖、下载官方相应模型后，运行 `bash inference/run.sh` 或 `python -m inference.run --input playground/case1/case1`。训练入口 `CONFIG_PATH=configs/<config>.yaml bash scripts/finetune/train.sh`；浏览器交互从 `reactor/` 单独部署。

## 实验观察与适用边界

- **作者报告：** 论文在 iWorld-Bench 上比较长程视觉质量、相机轨迹跟随和记忆一致性；论文称轨迹精度和记忆指标具竞争力。视频 24 FPS 指生成视频帧率，不能直接等价于消费级卡的实际生成吞吐。
- **论文/实现局限：** 部署涉及大权重、DA3/ViGeo 外部依赖与受门控的 Gemma 文本编码器，不能把“开放代码”解读为所有依赖即取即用。论文多版本 checkpoint 功能范围不同，不能混用；生成一致性不能证明符合真实碰撞动力学。

## 对 wiki 的映射

- [独立论文详情](../../wiki/entities/paper-alayaworld.md)
- [11 项开源世界模型技术对照](../../wiki/comparisons/open-source-world-models-11-2026.md)

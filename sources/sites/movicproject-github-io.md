# MoViC 官方项目主页（movicproject.github.io）

> 官方项目页来源归档；与 arXiv:2611.19998 为同一研究成果，不创建第二个项目实体。

- **URL：** <https://movicproject.github.io/>
- **论文：** <https://arxiv.org/abs/2611.19998>
- **项目名称：** MoViC: Real-Time Humanoid Whole-Body Control from Monocular Video Streams
- **页面标语：** One RGB camera · No wearable trackers · No per-video training
- **代码：** **待发布**；2026-11-18 页面顶部标记「Code(coming soon)」，没有官方 GitHub / 权重仓库的有效入口
- **数据：** 本次项目页未发现公开下载数据集入口，不能推断已开源
- **入库日期：** 2026-11-18
- **核对范围：** 已阅读主页的摘要、System Overview、Algorithm Framework、Real-World Experiments 与视频入口/代码状态，并与 arXiv HTML v1 方法、演示对应核对。
- **通读：** 已完成

## 页面信息与论文对应关系

| 项目页模块 | 一手信息 |
|---|---|
| 首页演示 | 单目实时模仿人体舞蹈、双臂搬箱、单臂提包 |
| System Overview | SAM 3D Body 单目重建约 10 Hz；机器人重定向与潜动力学外推生成 50 Hz 参考 |
| Algorithm Framework (a) | 冻结的频域 motion encoder 输出相位增强 θ 表征 |
| Algorithm Framework (b) | Transformer actor：本体历史 + 下一步参考 + θ；24 个 PD 关节目标 |
| Algorithm Framework (c) | SEG：单目重建、GMR 重定向、latent dynamics extrapolation |
| Algorithm Framework (d) | PPO + 训练期 privileged critic，不向 critic 输入 θ |
| Code / Resources | 仅「Code(coming soon)」；暂不能按源码验证运行入口 |

## 代码及复现状态

截至 2026-11-18，**论文方法与视频演示可公开核验，不等于官方代码或训练权重已公开**。因此不建立虚构的 `sources/repos/` 归档，也不绘制所谓“官方源码运行时序图”。后续若发布代码，应先核对 README、训练/推理脚本、许可证与对应论文版本，再补源码归档和真实时序图。

## 对 wiki 的映射

- [MoViC 论文来源](../papers/movic_arxiv_2611_19998.md)
- [MoViC 论文实体](../../wiki/entities/paper-movic-monocular-video-whole-body-control.md)

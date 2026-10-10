# WeiYuFei0217/StreamRig 官方代码仓库

- **类型：** repository / multi-camera visual odometry
- **仓库：** <https://github.com/WeiYuFei0217/StreamRig>
- **项目页：** <https://weiyufei0217.github.io/StreamRig/>
- **论文：** <https://arxiv.org/abs/2609.40244>
- **许可证：** CC BY-NC 4.0；仓库内 vendored MapAnything 保留 Apache 2.0，需分别遵守
- **一句话说明：** 提供 StreamRig 的训练与评测代码、NCLT / KITTI-360 数据预处理说明及发布权重入口。

## 仓库可复现入口

README 给出的主要流程：

1. 建立 Python 3.12 环境，安装 CUDA 12.8 对应的 PyTorch 2.9.1、requirements、vendored MapAnything 和 StreamRig。
2. 下载 MapAnything 的 DINOv2-Large 主干权重。
3. 分别生成数据集的 rig 元数据与冻结前端特征缓存。
4. 使用 G2G relocalization checkpoint warm-start，再通过 scripts/train.sh 训练 NCLT 或 KITTI-360 配置。
5. 使用 eval_nclt.sh / eval_kitti360.sh 和发布 checkpoint 复现 README 所列结果。

README 中的缓存体量很大：NCLT 约 470 GiB、KITTI-360 约 104 GiB；NCLT 评测约需 35 GB 主机内存。仓库提供的命令示例使用 4 张 GPU。实际环境要求应以当前仓库的配置和依赖版本为准。

## README 中的结果

| 数据集 | 测试序列 | t_rel（%） | r_rel（°/100 m） | ATE（m） |
|---|---|---:|---:|---:|
| NCLT | 2012-02-19、2012-08-20 | 2.77 | 1.39 | 28.4 |
| KITTI-360 | 0009、0010 | 2.59 | 0.98 | 63.7 |

仓库注明 t_rel / r_rel 使用 KITTI odometry 协议（100–800 m 分段、stride 3），ATE 在完整序列上做 SE(3) 对齐；录制中断位置按 README 所述用真值相对位姿衔接再计算。因此复现和横向比较时，应对齐序列、分段和对齐协议。

## 权重与许可

README 链接到 Hugging Face：<https://huggingface.co/feixue22/StreamRig>，另提供百度网盘；校验发布权重时可在 release_weights 目录使用 SHA256SUMS。代码为 CC BY-NC 4.0，不应作为可商用许可理解。

## 收录边界

项目页介绍四组评测数据（NCLT、TartanGround、KITTI-360、ZJH），但本仓库 README 目前明确列出的训练与评测脚本及公开权重表覆盖 NCLT、KITTI-360；其余数据的完整复现入口以仓库后续说明为准。

## 关联归档

- [StreamRig 论文题录与摘要](../papers/streamrig_arxiv_2609_40244.md)
- [StreamRig 官方项目页](../sites/streamrig-weiyufei0217-github-io.md)
- [StreamRig 论文 + 项目唯一节点](../../wiki/entities/streamrig.md)

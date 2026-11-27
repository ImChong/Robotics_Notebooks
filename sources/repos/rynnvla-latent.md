# RynnVLA-Latent 官方仓库

> 来源归档

- **论文：** <https://arxiv.org/abs/2610.34356v1>
- **项目页：** <https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io/>
- **代码：** <https://github.com/alibaba-damo-academy/RynnVLA-Latent>
- **核查提交：** [c67e6ff](https://github.com/alibaba-damo-academy/RynnVLA-Latent/tree/c67e6ff4f299f59459c5deb1c6f86b2451f5eec4)
- **仓库许可证：** Apache-2.0；外部训练数据与 benchmark 各有自己的条款。
- **核查日期：** 2026-11-27
- **通读：** 已完成
- **核对范围：** README、RynnLAM 训练与打标、VLA 调度/训练/导出、Stage-1/Stage-2 配置、LIBERO rollout 和推理封装；未运行本体推理、训练或 GPU 环境。

| 环节 | 官方入口 | 输出/接口 |
|---|---|---|
| 表示训练 | [train_rynnlam.py](https://github.com/alibaba-damo-academy/RynnVLA-Latent/blob/c67e6ff4f299f59459c5deb1c6f86b2451f5eec4/scripts/train_rynnlam.py) | 从预处理场景训练 RynnLAM |
| 离线标注 | [label_latent.py](https://github.com/alibaba-damo-academy/RynnVLA-Latent/blob/c67e6ff4f299f59459c5deb1c6f86b2451f5eec4/scripts/label_latent.py) | RynnLAMEncoder 输出 `ktoken_zcam`，写 `latent.npz`、帧对索引和来源字段 |
| 训练 | [launch.py](https://github.com/alibaba-damo-academy/RynnVLA-Latent/blob/c67e6ff4f299f59459c5deb1c6f86b2451f5eec4/rynnvla/api/launch.py) | torchrun 调用训练入口，构建数据集、模型和处理器；先 latent，再从 Stage-1 导出权重初始化原生动作训练 |
| 闭环 | [eval_libero.py](https://github.com/alibaba-damo-academy/RynnVLA-Latent/blob/c67e6ff4f299f59459c5deb1c6f86b2451f5eec4/rynnvla/api/eval_libero.py) | 导出的 Stage-2 目录 + 外部 LIBERO 安装；推理封装返回动作块，队列执行前若干步后规划 |

主仓自带两套很小的合成样例，供检查数据管道；不是完整训练语料。RoboTwin/VLABench 提供与外部框架对接的适配器，不等同于仓库自带完整平台和可测分检查点。

## 本次最小运行检查

在该提交用 CPU 跑官方 `scripts/smoke.sh --only data`，再对同一合成集执行 `scripts/verify_sample_data.py --num-samples 9`。生产数据加载器成功读取 3 个 episode 的全部 9 个样本，解码 64 × 64 RGB，输出有限的 `[6 个视角槽, 6 个时间步, 608]` 潜动作目标，样例仅有 1 个有效视角。未载入模型权重、未跑训练或 LIBERO。

- [论文归档](../papers/rynnvla-latent.md)
- [项目与模型发布](../sites/rynnvla-latent.md)
- [知识页](../../wiki/entities/paper-rynnvla-latent.md)

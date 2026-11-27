# NDM Gym：Neural Dynamics Modeling Gym

> 来源归档

- **类型：** repo（项目作者维护的官方代码与实验记录）
- **链接：** <https://github.com/Frank-ZY-Dou/ndm-gym>
- **项目文档：** <https://frank-zy-dou.github.io/ndm-gym/>
- **作者：** NDM Gym Contributors；仓库维护者 Zhiyang (Frank) Dou
- **维护归属：** 独立维护者与项目贡献者；官方引用署名 NDM Gym Contributors，未以单一机构名义发布。
- **核查版本：** [7c5f1fe](https://github.com/Frank-ZY-Dou/ndm-gym/tree/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c)
- **许可：** 代码 MIT；第三方轨迹、机器人模型及权重另看 [NOTICE](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/NOTICE) 和原始发布方许可。
- **核查日期：** 2026-11-27
- **沉淀到 wiki：** [NDM Gym](../../wiki/entities/ndm-gym.md)
- **通读：** 已完成
- **核对范围：** 固定提交的项目 README、覆盖清单、Quickstart、SuperTrack / SuperMimic / XMimic 与世界模型文档，以及 G1、双 G1、Steel、可微仿真、四旋翼、Adroit、Wuji、robomimic、RoboKit 和 mbrl-lib 的结果记录；交叉核对 G1 训练入口、命名配置、采集循环和两类策略更新。该项目以仓库作为引用对象，未将其中引用的独立论文归为 NDM Gym 论文。

## 原始资料入口

- [固定版本 README](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/README.md)；[覆盖清单](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/docs/coverage.md)，区分实现、移植选项、外部代码计时和进行中。
- [G1 入口](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/ndm_gym/train.py)、[命名配置](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/ndm_gym/configs.py)、[阶段计时与误差](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/benchmarks/g1_lafan1_tracking/results.md)。
- [仿真器吞吐与梯度](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/benchmarks/ndm_speed/results.md)、[Steel 配置与通过标准](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/benchmarks/steel/results.md)、[RoboKit](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/benchmarks/robokit_ndm/results.md)、[mbrl-lib](https://github.com/Frank-ZY-Dou/ndm-gym/blob/7c5f1fecb6a4b460dfe208c8bee6e994f811a72c/benchmarks/mbrl_lib/results.md)。

## 本次核查边界

- 以 CPU 执行固定提交的 `configs.expand`，验证 10 个 G1、双 G1 与四旋翼命名配置，以及列出配置与拒绝错用训练器两个行为。未启动 CUDA 仿真、重训或复现速度。

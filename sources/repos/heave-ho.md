# Heave-Ho — 官方 GitHub 仓库核对

- **仓库：** https://github.com/heaveho-humanoid/Heave-Ho
- **项目：** https://heaveho-humanoid.github.io/
- **论文：** https://arxiv.org/abs/2611.22275
- **核对日：** 2026-11-19
- **源码状态：** **待发布**。官方 README 明确写“plan to release the code by the end of December 2026”。截至核对日，仓库顶层仅 `README.md` 与 `assets/`；无可辨认的训练、推理、评测或真机部署入口，未见权重/数据下载入口。
- **核对范围：** 直接阅读官方仓库 README 的作者、机构、开源计划、Abstract、Overview、Citation；检查 GitHub 仓库顶层目录，未见实际算法实现。未执行代码（暂无可运行实现）。
- **通读：** 已完成

## 已公开的信息与未公开的信息

- **已公开：** 论文、项目主页与视频、README 技术概述、teaser/overview 图片。
- **待发布：** 多机 PPO、SONIC 残差训练与适配蒸馏、Qwen3-VL 规划器训练、语音识别与调度代码、部署/评测脚本、训练权重。以上是原文描述的系统模块，**不是**当前仓库存在的文件。
- **源码运行时序图：** 不适用；README 未提供可运行源码入口，不可编造函数或调用时序。
- **复现边界：** 现阶段只能按论文结构复核方法与结果，无法独立运行 HH-Bench 或复现真机指标。建议在官方正式发布代码后重新核对并增补真实运行时序图。

## 映射

- [论文归档](../papers/heave_ho_arxiv_2611_22275.md)
- [项目页归档](../sites/heave-ho.md)
- [唯一论文实体](../../wiki/entities/paper-heave-ho.md)

# DemoHLM 项目页

> 来源归档（site / BeingBeyond 官方项目页）

- **URL：** <https://beingbeyond.github.io/DemoHLM/>
- **镜像仓库：** <https://github.com/BeingBeyond/DemoHLM>（commit `cb913c1`，含项目主页与文档，无训练/控制源码）
- **关联论文：** `arXiv:2510.11258`（见 [demohlm_arxiv_2510_11258.md](../papers/demohlm_arxiv_2510_11258.md)）
- **核对范围：** 已核对 GitHub 镜像仓库 `BeingBeyond/DemoHLM @ cb913c1` 中 `README.md` 与 `docs/` 文件清单及官方 arXiv 全文，确认项目页展示了 10 项移动操作任务、Unitree G1 2-DoF 主动视觉颈部/夹爪改装及真机闭环对比视频说明；项目页未发布训练或推理脚本。
- **通读：** 已完成

## 核心要点

- 展示「单演示 -> 仿真三阶段轨迹合成 -> 模仿学习策略 -> Unitree G1 零样本 Sim2Real」完整闭环。
- 明确底层采用 AMO 50 Hz 全身控制器、高层采用 10 Hz 模仿学习策略（ACT/DP）及 RealSense D435 + FoundationPose++ 物体 6D 位姿感知。
- 截至 2026-11-21，项目页与关联 GitHub 仓库均未释出训练/部署代码与合成数据集。

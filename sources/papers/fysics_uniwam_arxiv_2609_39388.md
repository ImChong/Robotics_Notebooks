# UniWAM：混合流世界动作模型与操作锚点位姿

> 资料归档草稿；核查日期：2026-11-01。论文全文与源码尚未完成通读，不能视为完整 ingest。

- **论文：** [arXiv:2609.39388v1](https://arxiv.org/html/2609.39388v1)，2026-09-30 提交。
- **项目主页：** https://fysics-ai.github.io/uniwam.github.io/
- **官方代码：** https://github.com/Fysics-AI/UniWAM
- **导航权重：** https://huggingface.co/Fysics-AI/UniWAM
- **数据：** [MAP-Data](https://huggingface.co/datasets/Fysics-AI/MAP-Data)；[MAP-Bench 与评测轨迹](https://huggingface.co/datasets/Fysics-AI/MAP-Bench)
- **机构：** Fysics AI Physical Superintelligence Lab；复旦大学智能机器人与先进制造学院。

## 已核对的线索

- 论文提出导航与操作分流、共享视频和动作主干的世界动作模型；MAP 表示操作前底盘应到达的位置与朝向。
- 官方仓库提供模型训练与推理代码、MAP-Bench 导航权重与评测资料；README 明确真机任务训练权重未包含在发布包中。代码主体采用 Apache-2.0；其他组件、模型及数据许可需分别核对。
- MAP-Bench 训练与评测共享目标，但按起始位姿分组以避免相近视角泄漏；额外 MAP-Data 排除了基准目标。评测结果不等同于对未见目标的仿真泛化。
- 这项工作与 [UniWAM（arXiv:2610.02054）](./uniwam_arxiv_2610_02054.md) 同名但为不同论文，不应合并实体。

## 未完成

- 完成全文的方法、实验、失败案例与附录阅读，核对项目主页及模型和数据卡。
- 阅读官方运行入口及相关源码，创建统一 wiki 实体，并依据核实内容绘制项目流程图和源码运行时序图。
- 完成关联页、日志和仓库 CI 检查后，再将 PR 转为可审阅状态。

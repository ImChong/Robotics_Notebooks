# SuperMimic-Repro-Sim2Real

> 来源归档

- **类型：** repo（社区非官方复现）
- **链接：** <https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real>
- **核查版本：** [f8f60d7](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/tree/f8f60d7bbdba68468c050ef100af02d752ca238e)
- **原论文：** <https://arxiv.org/abs/2610.28480>
- **上游复现：** <https://github.com/CuboiLeo/SuperMimic-Reproduced>
- **许可证：** MIT；第三方数据与工具另有条款。
- **核查日期：** 2026-10-30
- **沉淀到 wiki：** [SuperMimic](../../wiki/entities/paper-supermimic.md)

## 开源与复现范围

作者说明训练算法基于上游社区复现，并增加大数据集训练、自定义舞蹈、视频转动作及 Unitree G1 部署；不是论文作者的官方实现。训练、评测和部署代码公开，Release 提供模型与真机视频；数据集需自行下载并遵守许可。v1.1 的 e24 模型只针对一段舞微调，其父模型未发布。

README 与 [METHOD.md](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/blob/f8f60d7bbdba68468c050ef100af02d752ca238e/docs/METHOD.md) 报告：严格论文观测设置下 LAFAN1 40 段归一化回报 0.56，论文对照 0.82；加入相对 anchor 位姿后可达 0.904，但该设置不可直接用于所述真机部署。BONES-SEED 的 e11c 全集报告 0.9305、46/129491 段失败。指标为作者自报，不同硬件与配置下不宜直接归因。

真机展示约 46 秒串联舞蹈；v1.1 的特定动作模型报告两次完成约 22 秒动作、使用松弛保护绳，之前九个模型尝试失败，未做消融。详见 [真机实验](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/blob/f8f60d7bbdba68468c050ef100af02d752ca238e/docs/REAL_ROBOT_LEAN35KFS.md)。

## 核查边界

已核对 README、METHOD 方法与结果部分、训练入口和部署 runner 的关键路径；原论文全文及其他必要文档、源码尚未通读，运行时序仍需逐项复核。本次未运行训练、真机或仓库 CI，相关结论仅作为待核实草稿。

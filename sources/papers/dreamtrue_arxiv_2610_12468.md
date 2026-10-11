# DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training

- **类型：** paper / robot world model / video prediction
- **版本：** arXiv:2610.12468v1，2026-10-08
- **论文：** [arXiv](https://arxiv.org/abs/2610.12468) · [HTML](https://arxiv.org/html/2610.12468)
- **项目页：** [DreamTrue](https://brave-eai.github.io/DreamTrue/)
- **代码：** [brave-eai/DreamTrue](https://github.com/brave-eai/DreamTrue)
- **数据/权重：** [ModelScope](https://modelscope.cn/datasets/huoxingdawang/DreamTrue)
- **代码状态：** 部分开源；官方仓库列出 calibration、wmvideo、reward 组件与部分数据，完整数据计划后续发布。
- **入库日期：** 2026-10-11

## 核心摘录

1. **动作条件几何校准：** 将动作轨迹渲染为图像空间条件，并离线校准其与视频坐标的对应关系，以改善动作—视频一致性。
   - **对 wiki 的映射：** [DreamTrue 实体](../../wiki/entities/paper-dreamtrue-action-faithful-world-model.md) 的方法栈。
2. **反事实后训练：** 修改记录动作并生成新动作/接触配置下的未来视频，以扩展稀缺失败交互覆盖；生成数据不是新采集的真实物理证据。
   - **对 wiki 的映射：** 实体页反事实扩增与风险。
3. **具身视频奖励：** 人工标注机器人、物体和交互缺陷视频训练奖励模型，再用于 RL 后训练。
   - **对 wiki 的映射：** 实体页监督链与 reward hacking 风险。
4. **报告结果：** AgiBot action-following 比较 SOTA；人评交互缺陷率 48.12% 降至 6.25%；作者报告 AgiBot World Challenge 2026 第一名。仅限论文协议。
   - **对 wiki 的映射：** 实体页评测表。
5. **开放范围：** 仓库公开 calibration、wmvideo、reward 组件与部分资产，注明完整数据后续发布。
   - **对 wiki 的映射：** 实体页工程实践。

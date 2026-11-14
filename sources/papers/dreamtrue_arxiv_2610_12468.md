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

## 全文通读复核（2026-11-13）

- **核对范围：** arXiv:2610.12468v2（2026-11-09）正文 §1–§5、必要附录 A–C（条件/训练、奖励标注、完整评测协议）；D–E补充示例说明，未观看视频。图表按文字层核对，参考文献浏览；浏览器项目页、三个模块README与world-model推理相关源码；未运行模型。
- **通读：** 已完成
- Wan2.1-VACE14B，101帧/三视角240×320；RGB和depth经VAE，mask与Plücker经3Dconv，沿宽拼视角。StageII冻结base/geometry/reward，仅更新DiT/VACE LoRA，rank128，约.7B可训参数。
- 反事实为固定起点、SE3扰动末端终点再插值/IK，过滤运动学不可达；无真实未来监督，不自动证明接触正确。录制动作才有PSNR reward(.5)，L1/L2/L3各1；Qwen3.5-9B reward评缺陷概率，不做nDTW训练奖励。
- AgiBot recorded800 episodes，反事实160条件×6模型=960视频×3人多数票。StageI→II交互缺陷48.12→6.25%、物体31.88→3.12%，不是机器人任务成功率；EWMScore-P反而72.84→72.51，不能称所有指标均升。
- policy outcome是240条固定策略动作回放，24组合各10次：SR MAE12.92→7.08pp，Spearman .937；不是在生成观测里重新查询策略的闭环部署。真值来自RoboTwin仿真，不是新增真机试验。
- calibration发布153666保留episodes/1660.31h，generator训练2232h另含仿真；reward原始44.9K视频、30.4K标签，过滤后31600训练+4893验证，不相加成生成模型训练视频数。
- 现公开推理/渲染/奖励接口与部分资产；wmvideo README明确无训练代码，完整校准pipeline和全量数据仍coming soon。与网站release文案需区分，完整StageII训练不可从公开推理入口复现。

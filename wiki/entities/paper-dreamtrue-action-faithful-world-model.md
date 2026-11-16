---
type: entity
project_id: dreamtrue-action-faithful-robot-world-model
tags: [paper, world-model, video-prediction, counterfactual-learning, reward-model, manipulation]
status: complete
updated: 2026-11-13
arxiv: "2610.12468"
project: https://brave-eai.github.io/DreamTrue/
code: https://github.com/brave-eai/DreamTrue
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../entities/paper-dreamwam.md
original:
  - ../../sources/papers/dreamtrue_arxiv_2610_12468.md
sources:
  - ../../sources/papers/dreamtrue_arxiv_2610_12468.md
  - ../../sources/sites/dreamtrue-project-page.md
  - ../../sources/repos/brave-eai-dreamtrue.md
summary: "DreamTrue 以离线几何校准对齐图像空间动作条件，通过反事实后训练扩大交互覆盖，并用具身视频奖励模型进行 RL 后训练；代码与部分数据已开，完整数据待发布。"
---

# DreamTrue：动作保真的机器人世界模型

## 一句话定义

**DreamTrue** 是面向机器人交互视频预测的世界模型框架：校准动作条件与视频运动的一致性，以反事实后训练扩展少见交互，再用具身视频奖励模型引导生成结果趋向物理合理。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| WM | World Model | 根据观测和条件动作预测环境未来的模型 |
| RL | Reinforcement Learning | 用奖励信号优化模型的后训练过程 |
| SOTA | State of the Art | 论文针对特定 benchmark 的比较结论 |

## 为什么重要

动作跟随和物体交互是两种检验：夹爪沿轨迹移动，不代表物体在失抓时也预测正确。本文先修复动作渲染的几何错位，再用缺陷反馈扩展失败接触覆盖，可用于理解世界模型为何会过度乐观。

## 方法栈

1. **动作几何校准：** 将机器人动作轨迹渲染成图像空间条件，以离线几何校准将动作表示与视频观测对齐。改善的是 action-to-video calibration，不自动保证所有接触动力学正确。
2. **反事实后训练：** 修改记录动作并生成扩展动作/接触配置的未来视频，补充真实数据少见的失败交互。合成结果不能当作真实机器人观测。
3. **具身视频奖励：** 用人工标注的机器人、物体和交互缺陷训练视频奖励模型，分数引导 RL 后训练。标注偏差和 reward hacking 是主要风险。

## 流程总览

```mermaid
flowchart LR
  data["视频与动作轨迹"] --> calib["图像空间动作条件\n几何校准"]
  calib --> wm["wmvideo 世界模型"]
  wm --> cf["反事实后训练"]
  cf --> reward["reward 模型\n人工缺陷标注"]
  reward --> rl["RL 后训练"]
  rl --> wm
```

本页以已通读的 **arXiv v2（2026-11-09）** 为准：Wan2.1-VACE14B三视角101帧，StageII冻结base/geometry/reward，只更新双LoRA。录制动作额外使用PSNR，反事实动作没有配对真值未来。

## 评测

| 指标 | 论文报告 | 说明 |
|---|---:|---|
| AgiBot action following | 对比 SOTA | 限论文 benchmark 设置 |
| 人评交互缺陷率 | 48.12% → 6.25% | 人工评估指标，需结合标注与采样协议 |
| AgiBot World Challenge 2026 | 第 1 名 | 作者报告名次，不替代独立复现 |

AgiBot recorded评测800 episodes；反事实160条件、每模型三位盲评者多数票。48.12→6.25%是StageI/II的交互缺陷率，非机器人成功率；综合EWMScore-P为72.84→72.51，并非全部指标提高。

策略结果评估使用240条固定动作回放、24组合，SR MAE12.92→7.08个百分点、Spearman .937；真值来自仿真，不是生成观测下重新查询策略的闭环测试。遮挡下视觉合理仍可能物理错误。

## 与其他工作对比

| 维度 | 普通条件视频模型 | DreamTrue |
|---|---|---|
| 动作条件 | 直接输入动作标签/token | 轨迹投影到图像空间并几何校准 |
| 失败覆盖 | 依赖已采集失败样本 | 反事实修改并生成候选未来 |
| 质量信号 | 视频/像素目标 | 人工缺陷视频奖励参与 RL 后训练 |

DreamTrue 是世界模型训练系统，不应等同于可直接部署的闭环 manipulation policy。

## 源码运行时序图

依据[固定提交的公开推理代码](../../sources/repos/brave-eai-dreamtrue.md)。完整StageII训练未发布，故不画虚构的RL训练调用。

```mermaid
sequenceDiagram
  participant SH as 三视角inference shell
  participant Demo as inference_demo_v2v_vace.py
  participant Data as dataset config与condition HDF5
  participant Model as WanV2VVaceInferenceModule
  participant Pipe as WanVideoPipeline
  SH->>Demo: Accelerate launch与本地模型路径
  Demo->>Data: 读取RGB首帧、动作渲染和相机参数
  Demo->>Model: 初始化base、enable_plucker、加载双LoRA
  Demo->>Model: generate三视角reference与条件
  Model->>Pipe: 多视角条件视频预测
  Pipe-->>Demo: 各视角未来帧
  Demo->>Demo: 裁padding、保存MP4与JSON记录
```

## 工程实践

- 对齐 `use_plucker` 与检查点geometry键；输入视角顺序、padding与相机参数必须匹配训练契约。
- 公开生成summary的success_rate仅指运行/文件生成成功，不能当任务SR。StageII训练代码与完整校准流程尚未提供，本次未复现。


- 先验证相机参数、动作坐标系和时间戳，避免动作条件错位。
- 按任务和接触类型检查反事实样本与真实数据差异。
- 人工缺陷奖励需做分组评估，防止均值掩盖安全失败。
- 官方仓库已部分开放，README 称完整数据后续发布；ModelScope 有资产不代表完整复现条件齐备。

## 结论

**结论：** DreamTrue 将动作—视频几何一致性作为世界模型训练的核心约束，但可信度仍取决于校准、合成反事实和奖励模型。

1. 先核对动作轨迹与图像坐标，再评估视频逼真度。
2. 反事实视频扩展覆盖，不构成真实物理验证。
3. 48.12%→6.25% 应与人评协议和样本规模一起报告。
4. 世界模型不等于闭环控制器或安全验证器。
5. 当前为部分开源，完整数据待补齐。

## 局限与风险

- 几何校准不能消除遮挡、柔性物体和复杂接触的动力学误差。
- 反事实生成可能产生逼真但物理错误的未来。
- 人工标签存在覆盖不足，奖励模型也可能被优化过程利用。
- 论文结果集中于特定 AgiBot 评测，需跨平台、真实闭环验证。

## 关联页面

- [World Action Models](../concepts/world-action-models.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [基于模型的强化学习](../methods/model-based-rl.md)
- [机器人操作](../tasks/manipulation.md)
- [DreamWAM](paper-dreamwam.md)

## 参考来源

- [论文摘录](../../sources/papers/dreamtrue_arxiv_2610_12468.md)
- [项目页归档](../../sources/sites/dreamtrue-project-page.md)
- [官方仓库归档](../../sources/repos/brave-eai-dreamtrue.md)
- [arXiv](https://arxiv.org/abs/2610.12468)

## 推荐继续阅读

- [arXiv v2全文](https://arxiv.org/html/2610.12468v2)：附录A–C给出模型、标注和固定动作回放协议。
- [官方wmvideo README](https://github.com/brave-eai/DreamTrue/tree/main/wmvideo)：推理所需资产与开放边界。

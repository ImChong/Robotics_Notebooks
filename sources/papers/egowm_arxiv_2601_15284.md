# EgoWM — Walk through Paintings: Egocentric World Models from Internet Priors

> 来源归档（paper）

- **标题：** Walk through Paintings: Egocentric World Models from Internet Priors（**EgoWM**）
- **类型：** paper
- **Venue：** ECCV 2026
- **arXiv：** <https://arxiv.org/abs/2601.15284>
- **项目页：** <https://egowm.github.io/>
- **代码：** <https://github.com/miccooper9/egowm>
- **权重：** <https://huggingface.co/anuragba/egowm/>
- **入库日期：** 2026-07-26
- **一句话说明：** 给任意预训练视频扩散模型加轻量动作条件层，把它变成 egocentric 动作条件世界模型；覆盖 3-DoF 移动到 25-DoF 人形关节轨迹，并提出 Structural Consistency Score（SCS）评结构一致性。

## 核心论文摘录（MVP）

### 1) 动机：用互联网视频先验，而不是从零训 WM

- **链接：** <https://arxiv.org/abs/2601.15284>
- **摘录要点：** 目标不是「看起来合理」的未来，而是 **随动作正确变化** 的未来。EgoWM 复用 Internet-scale 视频模型先验，经轻量 conditioning 注入电机命令，保留泛化与真实感，同时提高动作忠实度。
- **对 wiki 的映射：**
  - [EgoWM 实体页](../../wiki/entities/paper-egowm-egocentric-world-model.md)
  - [Generative World Models](../../wiki/methods/generative-world-models.md)

### 2) 方法与评测：跨本体 + SCS

- **链接：** <https://egowm.github.io/>
- **摘录要点：** 同一套路从 3-DoF 移动机器人扩到 25-DoF 人形关节角驱动动力学；支持导航与操作 rollout。引入 **SCS** 衡量稳定场景元素是否随动作一致演化；相对 Navigation World Models 等基线，SCS 最高可提升约 **80%**，延迟可低至约 **6×**。演示含绘画场景零样本导航与自采真机图泛化。
- **对 wiki 的映射：**
  - [WAM×运动控制五路径](../../wiki/overview/wam-motion-control-five-paths.md) — ⑤ 评估/表示外侧
  - [1XWM](../../wiki/entities/paper-1xwm-redwood-world-model.md) — 同属视频 WM，但 1XWM 侧重策略评测价值头

### 3) 开源状态

- **链接：** <https://github.com/miccooper9/egowm>
- **摘录要点：** **部分开源**：仓库已发布 SVD 导航推理脚本（3-DoF / 25-DoF nav）与权重下载说明；README TODO 标明 SCS 脚本、Wan2.1-14B / Cosmos 训练推理、SVD 训练与 25-DoF **操作** 推理仍为 Soon。数据依赖 RECON / SCAND / Tartan 与 1X raw video。
- **对 wiki 的映射：**
  - [miccooper9/egowm 仓库归档](../repos/miccooper9_egowm.md)

## 关键术语

- **EgoWM：** Egocentric World Model
- **SCS：** Structural Consistency Score — 与外观解耦的结构一致性指标

## 关联 Wiki 页面

- [paper-egowm-egocentric-world-model](../../wiki/entities/paper-egowm-egocentric-world-model.md)
- [wam-motion-control-five-paths](../../wiki/overview/wam-motion-control-five-paths.md)

## 当前提炼状态

- [x] arXiv / 项目页 / 代码 / HF 权重
- [x] 开源边界（部分）
- [x] wiki 映射

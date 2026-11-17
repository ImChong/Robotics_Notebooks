# Helix: A Vision-Language-Action Model for Generalist Humanoid Control

> 来源归档（RCL Awesome World-Action Models · ``docs/PAPERS.md`` / ``data/papers.json``）

- **列表：** [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- **PAPERS.md：** <https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md>
- **分组：** VLA
- **编号：** 239/564
- **入库日期：** 2026-09-25
- **arXiv：** （无 / 非 arXiv）
- **出处：** 2025
- **提交日：** —
- **论文：** <https://www.figure.ai/news/helix>
- **代码：** 未在清单中标注
- **项目页：** <https://www.figure.ai/news/helix>
- **子类 / 象限：** 分层与双系统VLA · 不适用
- **Contribution（清单）：** RCL Awesome WAM 清单收录（VLA）；细节以原文 PDF / 项目页为准。
- **沉淀到 wiki：** [`wiki/entities/paper-rcl-ref-cb61c489d1333f433fc4-helix-a-vision-language-action-model-for-general.md`](../../wiki/entities/paper-rcl-ref-cb61c489d1333f433fc4-helix-a-vision-language-action-model-for-general.md)

---

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 清单条目元数据 | 本 source 为策展摘录，非全文转存 |
| 代码/权重 | 以项目页 / GitHub 实际链接为准；清单标注见上 |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-rcl-ref-cb61c489d1333f433fc4-helix-a-vision-language-action-model-for-general.md`](../../wiki/entities/paper-rcl-ref-cb61c489d1333f433fc4-helix-a-vision-language-action-model-for-general.md)
- 列表实体：[`wiki/entities/paper-rcl-wam-robot-learning-control-survey.md`](../../wiki/entities/paper-rcl-wam-robot-learning-control-survey.md)


## 官方原文通读复核（2026-11-17）

- **核对范围：** Figure 2025-02-20 的 Helix 初代官方技术说明完整正文（Introducing Helix 至 Conclusion），包括模型与训练、流式推理、结果及讨论；图注文字已核对，视频画面未独立复现实验。该来源是技术博客，不是独立同行评审论文；本次未通读 Helix 02/2.5，也未读 ISO 标准全文。
- **通读：** 已完成
- **原文版本：** <https://www.figure.ai/news/helix>，读取日期 2026-11-17。

### 一手核对结果

- S2 是 7B 参数 VLM，以 7–9 Hz 处理图像、机器人状态与语言，输出连续语义 latent；S1 为 80M 参数视觉运动 Transformer，结合最新图像、状态与该 latent，以 200 Hz 输出 35-DoF 上半身动作。
- 输出是腕部位姿、手指屈伸/外展、头与躯干朝向等目标，另有任务完成百分比；200 Hz 是策略动作更新频率，不能写成电机电流环频率。
- 约 500 小时遥操作数据经 VLM 后验指令标注；训练时加入与部署推理时延匹配的 S1/S2 输入时间偏移。部署在两颗嵌入式 GPU 上，S2 异步更新共享 latent，S1 使用最新观测和 latent 闭环执行。
- 双机器人演示使用相同权重，支持同平台对象/任务泛化的案例；不能由此推出任意形态机器人无需适配，或证明所有公司都必须自研硬件。
- 官方正文描述新物体操作与协作演示，但未在该文提供完整逐任务重复试验成功率和安全认证证据；宣传中的“商业就绪”不等于独立验收。
- **源码/数据开放核查：** 本次检查完整页面的链接，未见 Helix 训练/部署源码、模型权重及上述训练数据的公开下载入口；所述 S2 骨干为开源模型不等于完整 Helix 已开源。源码运行时序图不适用。

### 关联编译页

- [具身大模型与本体协同设计](../../wiki/concepts/embodied-foundation-model-hardware-codesign.md)：用动作接口、异步控制和训练时延对齐说明策略与本体的协同约束。

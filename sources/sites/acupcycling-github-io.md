# Action Upcycling 项目页（acupcycling.github.io）

> 来源归档（site）

- **标题：** Don't Throw Away the Tail: Action Upcycling for Policy Acceleration
- **类型：** project-page
- **URL：** <https://acupcycling.github.io/>
- **论文：** [arXiv:2609.34911](../papers/action_upcycling_arxiv_2609_34911.md)
- **机构：** Sungkyunkwan University、KAIST
- **入库日期：** 2026-09-30
- **代码：** **已开源** — <https://github.com/star-kwon/action-upcycling>；归档 [`sources/repos/action-upcycling.md`](../repos/action-upcycling.md)
- **一句话说明：** 训练-free tail 复用：速度波动门控拉长 execution horizon，1.2–1.7× 减 policy call 且成功率不低于 baseline。

## 核查结论（步骤 2.5）

- **已公开：** 方法动画、LIBERO / LIBERO-Plus / RoboTwin 全表、与 AAC / AutoHorizon 对比、YAM 真机视频与分任务表
- **已开源：** 页头 **Code** 链至 `star-kwon/action-upcycling`（Apache-2.0）
- **Footer / 页内：** PDF（arXiv）、GitHub、真机结果锚点

## 页面要点摘录

- **Upcycling ratio：** \(r=\bar h/h\)；\(r=2\) 约减半 policy call
- **阈值：** 从策略自身 rollout 的 tail 信号池 \(\mathcal C\) 选最小 \(\tau\) 使 \(\bar h(\tau)\ge rh\)；**无额外 rollout、无成功标签**
- **互补：** 可与 few-step sampling、FlashVLA 等 **减单次 call 成本** 的方法叠加（论文报告最高约 6.7× 组合加速）
- **WAM：** FastWAM 上 **无需改架构** 即有效

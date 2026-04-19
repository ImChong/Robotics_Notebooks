# gentlehumanoid_upper_body_compliance

> 来源归档（ingest）

- **标题：** GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction
- **类型：** paper
- **来源：** arXiv HTML
- **原始链接：** https://arxiv.org/html/2511.04679
- **入库日期：** 2026-04-19
- **最后更新：** 2026-04-19
- **一句话说明：** 这篇论文把阻抗参考动力学、统一的弹簧式交互力建模和可调力阈值整合到 humanoid whole-body tracking policy 中，重点解决拥抱、搀扶站起、气球操作这类上半身密集接触任务中的柔顺控制问题。

## 核心摘录

### 1) GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction
- **链接：** <https://arxiv.org/html/2511.04679>
- **核心要点：**
  - 提出一个面向 humanoid 上半身柔顺交互的 whole-body control 框架，把 motion tracking 的 driving force 与来自人/物体接触的 interaction force 统一写进 reference dynamics。
  - interaction force 采用统一的 spring-based formulation，同时覆盖 **resistive contact**（机器人压在表面上时的恢复力）和 **guiding contact**（外界推/拉手臂时的引导力）。
  - 为了保证肩-肘-腕整条运动链的受力协调性，guiding contact 的 anchor 不是对每个 link 独立随机施力，而是从 human motion dataset 的完整 upper-body posture 中采样。
  - 训练时引入 **force thresholding**，在 5–15 N 区间内随机化安全阈值；部署时可以按任务调节柔顺程度，例如 5 N 用于握手/气球，10 N 用于拥抱，15 N 用于 sit-to-stand assistance。
  - 使用 teacher-student PPO 进行 sim-to-real；student 只依赖真实可得观测，teacher 额外看 privileged reference dynamics / interaction force / torque 等信息。
  - 论文在 Unitree G1 上验证了静态受力、mannequin hugging、balloon handling 等任务，并给出 40 个 calibrated capacitive taxels 的压力垫评测。
  - 论文还展示了两条面向实际部署的扩展链路：一条是 **shape-aware autonomous hugging**（结合 BEDLAM 支撑的人体 shape 估计），另一条是 **video-to-humanoid**（用 PromptHMR 从手机单目 RGB 视频恢复 SMPL-X motion，再经 GMR 重定向到 G1）。
- **关键结论：**
  - 这篇工作不是单纯让机器人“更软”，而是让它在 whole-body tracking 成功率和 interaction safety 之间获得可调、可部署的平衡。
  - 对 Robotics_Notebooks 来说，它是连接 **Whole-Body Control / Contact Dynamics / Loco-Manipulation / Human-Robot Interaction** 的一篇非常典型的桥接论文。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [Contact Dynamics](../../wiki/concepts/contact-dynamics.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

## 当前提炼状态

- [x] HTML 原文核心方法与实验信息已摘录
- [x] wiki 页面映射确认
- [ ] 相关 wiki 页面的参考来源段落已补 ingest 链接
- [ ] 若后续需要，可继续拆成 upper-body compliance / safe physical interaction / assistive humanoid interaction 子页面

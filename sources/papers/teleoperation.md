# teleoperation

> 来源归档（ingest）

- **标题：** 遥操作与人形机器人数据采集核心论文
- **类型：** paper
- **来源：** RSS / CoRL / arXiv / Science Robotics
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 覆盖 ALOHA/ACT 遥操作系统、OmniH2O 全身遥操作、UMI 通用操作接口等

## 核心论文摘录

### 1) Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（Zhao et al., RSS 2023）
- **链接：** <https://arxiv.org/abs/2304.13705>
- **核心贡献：** ALOHA 遥操作系统：4 个低成本机械臂（$20K 总成本）+ shadow arm 设计；采集精细双臂操作示范；配合 ACT 策略学习；在精细任务（开包装/折衣物）成功率超过 50%
- **关键洞见：** 遥操作系统设计关键：人机运动对应 + 末端执行器映射 + 低延迟传输
- **对 wiki 的映射：**
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md)
  - [motion-retargeting](../../wiki/concepts/motion-retargeting.md)

### 2) OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation（He et al., 2024）
- **链接：** <https://arxiv.org/abs/2406.08858>
- **核心贡献：** 全身人-人形遥操作：VR 头显 + 手套捕获人体全身运动；实时重定向到 H1/G1 人形机器人；同时控制移动基座 + 双臂；展示遥操作数据用于 RL 微调的完整 pipeline
- **关键洞见：** 遥操作数据 → RL fine-tuning → 真机部署是当前人形最可行路线
- **对 wiki 的映射：**
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md)
  - [motion-retargeting](../../wiki/concepts/motion-retargeting.md)

### 3) Universal Manipulation Interface（Chi et al., RSS 2024）
- **链接：** <https://arxiv.org/abs/2402.10329>
- **核心贡献：** UMI：手持夹爪 + GoPro 相机的通用操作数据采集设备；无需专用机器人即可采集示范；数据直接用于 Diffusion Policy 训练；跨机器人形态迁移（夹爪 → 机械臂）
- **关键洞见：** 数据采集工具的通用性决定数据规模 → 数据规模决定策略泛化能力
- **对 wiki 的映射：**
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md)
  - [motion-retargeting](../../wiki/concepts/motion-retargeting.md)

### 4) Expressive Whole-Body Control for Humanoid Robots（Cheng et al., RSS 2024）
- **链接：** <https://arxiv.org/abs/2402.16796>
- **核心贡献：** 基于 RL + 运动重定向实现全身表情控制；从 SMPL 人体姿态序列重定向到人形机器人；支持实时遥操作和离线动作回放；在 H1 实现自然的人类动作模仿
- **对 wiki 的映射：**
  - [motion-retargeting](../../wiki/concepts/motion-retargeting.md)
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md)

### 5) AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System（Wei et al., RSS 2023）
- **链接：** <https://arxiv.org/abs/2307.04577>
- **核心贡献：** 仅用 RGB 相机的通用手臂遥操作系统；不需要专用传感器（无手套/标记点）；跨机器人形态迁移；基于视觉手势估计的低成本遥操作
- **对 wiki 的映射：**
  - [motion-retargeting](../../wiki/concepts/motion-retargeting.md)
  - [loco-manipulation](../../wiki/tasks/loco-manipulation.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [ ] 关联 wiki 页面的参考来源段落已添加 ingest 链接

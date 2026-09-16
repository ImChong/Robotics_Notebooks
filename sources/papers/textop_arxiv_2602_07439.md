# textop_arxiv_2602_07439

> 来源归档（ingest）

- **标题：** TextOp: Real-time Interactive Text-Driven Humanoid Robot Motion Generation and Control
- **类型：** paper
- **来源：** arXiv:2602.07439（2026-02-06 预印本）
- **作者：** Weiji Xie, Jiakun Zheng, Jinrui Han, Jiyuan Shi, Weinan Zhang, Chenjia Bai, Xuelong Li
- **机构：** 中国电信人工智能（TeleAI）、上海交通大学（SJTU）、华东理工大学（ECUST）
- **入库日期：** 2026-09-16
- **最后更新：** 2026-09-16
- **项目页：** <https://text-op.github.io/>
- **代码：** <https://github.com/TeleHuman/TextOp>（MIT；含预训练权重与数据集脚本）
- **一句话说明：** 实时流式文本驱动人形全身控制：高层自回归运动扩散（VAE+LDM，\(T_{\mathrm{hist}}=2,T_{\mathrm{fut}}=8\)）+ 低层 Isaac Lab RL 通用跟踪；G1 真机「一镜到底」多技能切换，用户交互延迟约 **0.73 s**；**已开源**。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2602.07439>
- **痛点：** 通用人形跟踪器已能执行多样全身动作，但驱动方式仍多为 **预录轨迹**（意图难改）或 **持续遥操作**（人力成本高）；语言→运动工作多为 **离线整段生成** 或 **开环级联**，缺少执行中 **随时改令** 的交互闭环。
- **TextOp 主张：** 两层架构——高层 **自回归文本条件运动扩散** 持续产出短视界 kinematic 参考；低层 **通用运动跟踪策略** 在真机上高频执行；语言与运动均视为 **时变流** 而非一次性规划。
- **对 wiki 的映射：**
  - [TextOp 实体](../../wiki/entities/paper-loco-manip-161-022-textop.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

### 2) 数据与机器人骨架表征（§III-B / §III-C）

- **数据：** AMASS + 少量私有舞蹈/武术；GMR 重定向到 G1；特权跟踪器过滤不可跟踪片段 → **12,296** clip（40.67 h）+ 私有 **403** clip（3.12 h）。
- **语言：** BABEL 帧级标注 + 镜像增广 → **83,478** 段–文本对；私有长动作用 `⟨·⟩` 包裹唯一标签。
- **表征：** 相对 HumanML3D / RobotMDM，提出 **DoF 局部增量机器人骨架特征**（根姿态三角编码、接触、关节位姿与增量等），更贴合单 DoF 关节结构。
- **生成器：** DART 式 VAE + 潜空间 LDM（Transformer）；CLIP 文本嵌入；DDPM **5** 步去噪；CFG scale **5**；自回归 self-rollout 减 train–deploy 分布差。
- **对 wiki 的映射：**
  - [Diffusion Motion Generation](../../wiki/methods/diffusion-motion-generation.md)
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)

### 3) 跟踪与生成器数据增广（§III-D）

- **跟踪器：** Isaac Lab 单阶段 PPO，MLP 策略；参考窗 \(T_{\mathrm{ref}}=5\)；奖励与域随机化主要沿 BeyondMimic。
- **增广：** 从 BABEL 采样 20 s 文本流，经高层生成器合成 **5,368** clip（31.48 h），与动捕混合训练 **TextOp-M+G**，对齐部署时生成器输出分布。
- **对 wiki 的映射：**
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 4) 部署（§III-E）

- **平台：** Unitree G1（29 DoF）；跟踪策略机载 **50 Hz**（ONNX Runtime）；生成器外置工作站 **6.25 Hz**（RTX 4090 + TensorRT）；网络 + motion buffer 同步。
- **延迟：** 文本编码 **7.64 ms**；生成器 **29.63 ms**；跟踪 **2.15 ms**；**用户交互延迟 0.73±0.10 s**（打字到新指令的物理响应）。
- **对 wiki 的映射：**
  - [textop 仓库](../repos/textop.md)

### 5) 实验数字（§IV）

**真机 30 s 长时域（表 I）：**

| 文本流 | Succ | \(E_{\mathrm{g-mpjpe}}\) | \(E_{\mathrm{mpjpe}}\) |
|--------|------|---------------------------|------------------------|
| Random | 16/20 | 337.2 | 153.9 |
| Looping "punch" | 10/10 | 355.6 | 123.6 |
| "wave right hand" | 8/10 | 316.5 | 72.2 |
| "strum guitar…" | 10/10 | 191.6 | 88.7 |
| "play the violin" | 10/10 | 107.6 | 33.9 |

**生成质量（BABEL val，表 III，节选）：** TextOp 段级 FID **3.072**、R@1 **0.300**、过渡 FID **3.238**、AUJ **0.125** — 优于 DART+Retarget、HumanML3D、RobotMDM 等表征基线。

**跟踪（生成器产出参考，Sim2Sim 表 IV）：** TextOp-M+G Succ **0.993**、\(E_{\mathrm{mpjpe}}\) **34.7 mm**；相对 GMT / Any2Track / TWIST2 显著更稳。

**局限（§V）：** 无环境感知与交互物理推理，不能绕障或操作物体。

- **对 wiki 的映射：**
  - [SafeFlow](../../wiki/entities/paper-loco-manip-161-104-safeflow.md)（相对 TextOp 的物理安全对照）
  - [ADAPT](../../wiki/entities/paper-adapt-text-driven-humanoid.md)（端到端扩散对照；Offline TextOp Success 0.522）

### 6) 开源核查（步骤 2.5，2026-09-16）

| 项 | 状态 |
|----|------|
| 项目页 | <https://text-op.github.io/> — 摘要、演示视频、BibTeX |
| GitHub | <https://github.com/TeleHuman/TextOp> — **已开源**（MIT）；`TextOpRobotMDAR` / `TextOpTracker` / `TextOpDeploy`；README 提供预训练权重与数据集处理脚本 |
| 备注 | 2026-02 官方版发布；README 注明「最新代码与数据集尚未全部更新」，但训练/部署入口可辨识 |
| 结论 | **已开源** — 可复现高层扩散、低层跟踪与 sim2sim/sim2real 部署 |

## 其他公开资料

- 演示视频：<https://youtu.be/nKxE7ff1FwY>
- 深读笔记（Robot_Learning_Paper_Notebooks）：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/TextOp__Real-time_Interactive_Text-Driven_Humanoid_Robot_Motion_Generation_and_C/TextOp__Real-time_Interactive_Text-Driven_Humanoid_Robot_Motion_Generation_and_C.html>

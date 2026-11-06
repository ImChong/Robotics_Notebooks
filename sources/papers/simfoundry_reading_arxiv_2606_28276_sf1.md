# SimFoundry 论文通读（Real2Sim 场景生成视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-simfoundry-real2sim-scene-generation.md`）

- **标题：** SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation
- **作者：** Nadun Ranawaka, Josiah Wong 等（NVIDIA / Georgia Tech / Stanford / UT Austin / Toronto）
- **类型：** paper / real2sim / sim2real
- **原文：** <https://arxiv.org/abs/2606.28276>（2026-11 通过 arXiv PDF v4，2026-08-05 版获取全文，52 页）
- **一句话说明：** 从单段 RGB 视频自动重建 sim-ready 数字孪生，并生成 object / scene / task 三类 digital cousins，同时用于 real-to-sim 策略评测与 sim-to-real 训练。
- **核对范围：** 通读正文 §1–§7（含 Table 1、Fig.4–5、Table 2）及附录 B（FAQ）、C（局限）、E（提取 / 基础模型 / 铰接 / 去穿透 / 背景 splat）、F（三类 cousins 流程与 prompt）、G（Table G.1/G.2/G.4 与 cousin 消融）、H（数据生成与训练细节）、I（机器人平台）、J（评测协议、指标、PolaRiS 对照）、K（人工交互）、L（重建评测与背景管线对比）。附录 D 相关工作综述、各任务打分细则（I.2）、各 prompt 模板全文与 Table L.5/L.6 的逐项数值仅浏览，未逐条核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **三阶段（§4）：** Extraction（DepthAnything3 估深度，SAM3 分割，Gemini 检物，迭代 inpaint 直到前景清空，默认取视频第 0 帧）→ Generation（Hunyuan3D-2.1 或 TRELLIS.2 出 mesh，FoundationPose 精化位姿，CoACD 碰撞体，VLM 标质量 / 摩擦，PyBullet 沉降去穿透后导出）→ Augmentation（三类 cousins）。
- **铰接（附录 E.3）：** VLM 列出可动部件与关节类型，P3-SAM 等分割 mesh（会过分割，再由 VLM 合并），借鉴 Articulate-Anything 的 actor-critic 生成 URDF 关节，critic VLM 看仿真回放打分迭代。
- **局限（附录 C）：** 依赖第三方 VLM，输出不确定（Gemini inpaint 偶有重复 / 退化物体）；单目点云尺度可能偏；沉降假设物体放在同一平面，限于桌面布局；自动背景管线双遍视频 inpaint 约 90 分钟 / 场景；部分机器人实验因 3DGS 近场裁剪与渲染延迟改用 mesh 背景。
- **Cousins（§4、附录 F）：** object cousins 经 VLM 按抓取 affordance 拆部件、在几何 / 拓扑 / 外观三维提议变体，图像模型编辑后重生成 mesh，并经 VLM 合理性过滤；scene cousins 用 LeftOf / RightOf / OnTop 等谓词重排并加 distractor；task cousins 由 VLM 看场景图加机器人约束提议需要状态变化的任务并编译为谓词目标。
- **评测（§5.1，Table G.1）：** 7 任务 × 5 策略（π0、π0.5、GR00T N1.6/N1.7、DreamZero）；每任务 25 次 rollout、5×5 初始位置网格；均值 Pearson 0.911、MMRV 0.018；子任务评测把均值 Pearson 由 0.90 提到 0.95；PolaRiS 在同协议下大幅更低（部分任务为负相关，Table G.2），其较好结果依赖在仿真数据上浅微调，本文评测不对策略做仿真适配。
- **训练（§5.2，附录 H）：** 约 10–15 条人工 JoyLo 演示经 MimicGen 扩增并做材质 / 相机 / 桌高随机化；DROID 上微调 π0 / π0.5（batch 256，lr 1e-5，10k 步），YAM 上从零训 flow-matching 策略；按仿真评测挑 checkpoint 再上真机。Pot on Stove 99%、Stack Dishware 100%；Store Marker co-train 60%→92%。
- **Cousins 消融（Fig.5、Table G.4）：** 平均提升 object +17%、scene +21%（twin 场景约 +13%，cousin 场景约 +29%）、task +40%；Pot on Stove 留出物体真机 +50 点；Store Marker cousin 布局 0%→16%。
- **FAQ（附录 B）：** DROID 预训练 π0 / π0.5 在 Store Marker、Throw Away Trash 上为 0%，Stack Dishware 为 52% / 48%，仅仿真微调后达 100%；采用二值成功率而非归一化奖励，因更严格。
- **多任务（Table 2）：** 13 个生成任务 + 7 个留出任务；π0.5-DROID 真机 28%，π0.5-DROID-FT 46%，π0.5-FT 留出任务真机 29%。
- **重建精度（§5.3，附录 L）：** 12 个 YCB 桌面场景（按遮挡分易 / 中 / 难）；零样本 F1 0.92 / 0.87 / 0.81，对 SAM3D 为 0.71 / 0.66 / 0.68；每物体约 3 分钟人工微调后 0.99 / 0.97 / 0.93；平均约 5 分钟 / 物体。
- **背景（附录 E.5、L.2）：** 自动管线（前景擦除 + 深度监督 splat）与手动二遍拍摄管线产出相同资产结构；自动管线可能在支撑面附近出现 inpaint 幻觉造成的漂浮物，但在 5 场景的渲染相似度指标上优于手动对齐。

## 对 wiki 的映射

- [`wiki/entities/paper-simfoundry-real2sim-scene-generation.md`](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2606.28276>

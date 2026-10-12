# 机器人会操作，为什么仍听不懂空间指令？（具身智能小站 · 34 篇）

> 来源归档（blog / 微信公众号）

- **标题：** 机器人会操作，为什么仍听不懂空间指令？
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/HSsx-QeViQjmnsOTL_FU2Q
- **发表日期：** 2026-10-12
- **入库日期：** 2026-10-12
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA；WebFetch/Jina 对公众号返回验证页）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md`](../raw/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md)
- **一句话说明：** 当日 34 篇具身新论文盘点，按阅读优先级分 1 篇深读 / 2 篇跟进 / 31 篇扫读；**34/34 独立详情节点**（本批 **新建 34**、**复用 0**）。

## 34 篇 → 本库节点

| # | 档位 | 论文 | arXiv | 文章中的一句话判断 | wiki |
|---|------|------|-------|--------------------|------|
| 01 | 先读这一篇 | Chronos Policy | [2610.13929](https://arxiv.org/abs/2610.13929) | 让机器人策略显式理解物理时间 | [Chronos Policy：在物理时间上学习机器人策略](../../wiki/entities/paper-chronos-policy-physical-time.md) |
| 02 | 值得跟进 | Spa-Bench | [2610.12696](https://arxiv.org/abs/2610.12696) | 会操作，不代表理解空间指令 | [Spa-Bench：真机上检验 VLA 的空间落地](../../wiki/entities/paper-spa-bench-spatial-grounding-vla.md) |
| 03 | 值得跟进 | LeWAM | [2610.12818](https://arxiv.org/abs/2610.12818) | 用同一个潜在模型执行与规划 | [LeWAM：用同一个潜在模型执行与规划](../../wiki/entities/paper-lewam.md) |
| 04 | 扫读 | Evo-RL | [2610.13345](https://arxiv.org/abs/2610.13345) | 用有价值的经验持续改进机器人策略 | [Evo-RL：从有选择的经验中迭代改进真机策略](../../wiki/entities/paper-evo-rl.md) |
| 05 | 扫读 | SmolWAM | [2610.13740](https://arxiv.org/abs/2610.13740) | 把世界动作模型压缩到更低推理成本 | [SmolWAM：把世界-动作模型压到边缘设备](../../wiki/entities/paper-smolwam.md) |
| 06 | 扫读 | RARA | [2610.13578](https://arxiv.org/abs/2610.13578) | 少量示范微调时保住原有泛化能力 | [RARA：检索锚定的表征对齐微调](../../wiki/entities/paper-rara-retrieval-anchored-finetuning.md) |
| 07 | 扫读 | Human2Dex | [2610.13119](https://arxiv.org/abs/2610.13119) | 同一份人类示范服务不同灵巧手 | [Human2Dex：腕部视角人类示范驱动多只灵巧手](../../wiki/entities/paper-human2dex-wrist-centric-dexterous.md) |
| 08 | 扫读 | Forward Dynamics Models | [2610.13117](https://arxiv.org/abs/2610.13117) | 用未来监督动作，部署时省去未来生成 | [FDM：动作到未来的前向动力学策略](../../wiki/entities/paper-fdm-forward-dynamics-models.md) |
| 09 | 扫读 | JitterVLA | [2610.13628](https://arxiv.org/abs/2610.13628) | 在同一命令流上比较机器人运动平滑性 | [JitterVLA：VLA 人机操作的运动质量回放基准](../../wiki/entities/paper-jittervla-replay-benchmark.md) |
| 10 | 扫读 | Grounded-Dreamer | [2610.13164](https://arxiv.org/abs/2610.13164) | 用物理验证筛选世界模型合成示范 | [Grounded-Dreamer：带物理验证的世界模型合成示范](../../wiki/entities/paper-grounded-dreamer.md) |
| 11 | 扫读 | Evo-World | [2610.12936](https://arxiv.org/abs/2610.12936) | 用光流监督轻量策略的未来运动预测 | [Evo-World：用光流监督的轻量 VLA 未来运动预测](../../wiki/entities/paper-evo-world-optical-flow-vla.md) |
| 12 | 扫读 | VLMs are Native Action Models | [2610.12832](https://arxiv.org/abs/2610.12832) | 让视觉语言模型直接看懂动作坐标系 | [EASG：把动作坐标系画进图像，让 VLM 直接当动作模型](../../wiki/entities/paper-easg-vlm-native-action-model.md) |
| 13 | 扫读 | Fly-by-Code | [2610.12549](https://arxiv.org/abs/2610.12549) | 让飞行操作智能体主动查明失败原因 | [Fly-by-Code：让空中操作的编码智能体主动查明失败原因](../../wiki/entities/paper-fly-by-code-aerial-manipulation.md) |
| 14 | 扫读 | PanTact | [2610.12558](https://arxiv.org/abs/2610.12558) | 让人形机器人用身体接触完成操作 | [PanTact：用可穿戴接触接口教人形机器人全身操作](../../wiki/entities/paper-pantact-whole-body-contact-interface.md) |
| 15 | 扫读 | Evidence-Guided Counterfactual Learning for Long-Tailed Robotic Policies | [2610.12861](https://arxiv.org/abs/2610.12861) | 减轻长尾任务被高频动作模式压制 | [ECL：证据引导的反事实学习，缓解 VLA 长尾任务被头部动作压制](../../wiki/entities/paper-ecl-evidence-guided-counterfactual-long-tail.md) |
| 16 | 扫读 | AutoBio+ | [2610.13643](https://arxiv.org/abs/2610.13643) | 把实验室动态现象纳入机器人评测 | [AutoBio+：带动态实验现象的生物实验室灵巧操作基准](../../wiki/entities/paper-autobio-plus-lab-dexterous-benchmark.md) |
| 17 | 扫读 | Non-Prehensile Multi-Object Placement with Mobile Robots via Diffusion-Based Trajectory Planning | [2610.12684](https://arxiv.org/abs/2610.12684) | 用因果扩散规划精准推放多个物体 | [Causal Diffusion Planner：移动机器人精确推放多个物体](../../wiki/entities/paper-causal-diffusion-nonprehensile-multi-object-placement.md) |
| 18 | 扫读 | REMM | [2610.12710](https://arxiv.org/abs/2610.12710) | 扰动后撤回过时结论，再安排修复 | [REMM：给冻结操作策略加“可修订记忆”](../../wiki/entities/paper-remm-revisable-embodied-memory.md) |
| 19 | 扫读 | Beyond Successful Robot Demonstrations | [2610.13522](https://arxiv.org/abs/2610.13522) | 世界模型也需要学习失败的交互动力学 | [HF-WM：向人类交互学习并建模失败动力学的世界模型](../../wiki/entities/paper-hf-wm-failure-dynamics-world-model.md) |
| 20 | 扫读 | Robot-WATCHDOG  | [2610.13403](https://arxiv.org/abs/2610.13403) | 用对象关系图识别操作失败 | [Robot-WATCHDOG：用对象关系图检测操作失败](../../wiki/entities/paper-robot-watchdog-object-graph-failure-detection.md) |
| 21 | 扫读 | CASTER | [2610.12567](https://arxiv.org/abs/2610.12567) | 把生成视频转为受约束的机器人轨迹 | [CASTER：把生成视频当作约束优化的初值](../../wiki/entities/paper-caster.md) |
| 22 | 扫读 | TAME-Nav | [2610.12877](https://arxiv.org/abs/2610.12877) | 通过时序专家蒸馏适应复杂导航场景 | [TAME-Nav：时序非对称 MoE 蒸馏的敏捷导航](../../wiki/entities/paper-tame-nav.md) |
| 23 | 扫读 | StereoEngine | [2610.12734](https://arxiv.org/abs/2610.12734) | 没有双目相机，也能扩充双目示范 | [StereoEngine：没有双目相机也能扩充双目示范](../../wiki/entities/paper-stereoengine.md) |
| 24 | 扫读 | PriMoRetarget | [2610.12642](https://arxiv.org/abs/2610.12642) | 用优先级保住动作迁移中的关键接触 | [PriMoRetarget：用优先级层级保住关键接触的人形动作重定向](../../wiki/entities/paper-primoretarget.md) |
| 25 | 扫读 | AlphaNIST | [2610.13354](https://arxiv.org/abs/2610.13354) | 只用二元奖励学习接触丰富的装配 | [AlphaNIST：只用二元奖励学会 NIST 装配](../../wiki/entities/paper-alphanist.md) |
| 26 | 扫读 | What Should Embodied Agents Remember? Benchmarking and Evaluating Memory Modality for Embodied Agents | [2610.12561](https://arxiv.org/abs/2610.12561) | 比较机器人该存文字、图像还是场景图 | [MeMoREval：具身智能体该记文字、图像还是场景图？](../../wiki/entities/paper-memoreval-embodied-memory-modality.md) |
| 27 | 扫读 | Beyond Actions | [2610.13212](https://arxiv.org/abs/2610.13212) | 让策略预测动作将产生的物理效果 | [AFCE：让 VLA 预测「动作将产生的效果」而不只是动作](../../wiki/entities/paper-afce-action-future-coupled-effect-vla.md) |
| 28 | 扫读 | Belief-Overlap Partial Inversion | [2610.13087](https://arxiv.org/abs/2610.13087) | 根据新旧信念的一致程度修改推车计划 | [BOPI：按新旧信念的一致程度修订人形推车计划](../../wiki/entities/paper-bopi-belief-overlap-partial-inversion.md) |
| 29 | 扫读 | MARGoS | [2610.13772](https://arxiv.org/abs/2610.13772) | 把群体强化学习复现变成平台职责 | [MARGoS：把群体强化学习的复现变成平台职责](../../wiki/entities/paper-margos-reproducible-marl-argos.md) |
| 30 | 扫读 | ATHLETE | [2610.13572](https://arxiv.org/abs/2610.13572) | 从普通视频学习反应式人形网球 | [ATHLETE：用通用行走数据加普通视频学人形反应式网球](../../wiki/entities/paper-athlete-reactive-humanoid-tennis-tgmm.md) |
| 31 | 扫读 | LeapBot-JAC | [2610.12886](https://arxiv.org/abs/2610.12886) | 在固定动作词元预算内安排两条动作链 | [LeapBot-JAC：世界-动作模型里的两条联合动作链](../../wiki/entities/paper-leapbot-jac-joint-action-chains.md) |
| 32 | 扫读 | The value of information and the emergence of communication in evolved e-puck robots | [2610.13118](https://arxiv.org/abs/2610.13118) | 通信何时对群体机器人真正有用 | [e-puck 通信价值演化：信号只在带来新信息时出现](../../wiki/entities/paper-epuck-value-of-information-communication.md) |
| 33 | 扫读 | Multimodal Image-based Robotic Assistance and Guidance Environment (MIRAGE) | [2610.12780](https://arxiv.org/abs/2610.12780) | 把多模态手术导航集成到机器人控制台 | [MIRAGE：达芬奇控制台里的统一手术导航与可视化框架](../../wiki/entities/paper-mirage-davinci-surgical-navigation.md) |
| 34 | 扫读 | Event-based Action Recognition by Pre-Training on Multi-Frame Representations | [2610.13856](https://arxiv.org/abs/2610.13856) | 用多帧预训练识别长时段事件动作 | [事件相机动作识别：多帧表示上的 MAE 预训练](../../wiki/entities/paper-event-action-recognition-multi-frame-mae.md) |

## 说明

- 文章给出的“代码 / 项目页”链接只作线索：各详情节点已逐一打开核对，部分链接实为 HF 数据集或匿名镜像，开源结论以节点为准。
- 文章的“看点 / 结果”是编辑摘要；成功率等数字请回到各节点引用的 arXiv 原文。

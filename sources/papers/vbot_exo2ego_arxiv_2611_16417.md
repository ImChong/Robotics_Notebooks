# VBot Exo2Ego Policy：轻量采集与第三人称到第一人称视频生成

> 一手论文的结构化核读归档；不转存论文全文，数字均以 arXiv v1 原文表格/附录为准。

- **论文：** https://arxiv.org/abs/2611.16417
- **原文 HTML：** https://arxiv.org/html/2611.16417 （arXiv:2611.16417v1）
- **PDF：** https://arxiv.org/pdf/2611.16417
- **标题：** VBot Exo2Ego Policy: Lightweight Data Collection and Exocentric-to-Egocentric Video Generation
- **作者：** Tong Rao、Zhihao Zhang、Yuntian Wang、Siyu Zhou、Hailong Qin、Yinan Yu
- **机构：** 维他动力（Vbot Physical AI Lab，Shanghai）
- **发表：** 2026-11-12（arXiv v1，cs.CV）
- **入库：** 2026-11-16
- **项目/代码：** 截至入库日，论文 arXiv 页面与全文未提供独立官方项目主页、训练/推理代码仓库或可运行入口；本轮未验证源码、权重和数据集开放。
- **核对范围：** 通读 arXiv v1 HTML：摘要、§1–7（数据采集 §3、生成方法 §4、手–物辅助监督 §5、实验 §6、局限 §7）、Table 1–7、Fig.1–4，附录 A（训练与评测 A.1–A.6）、B（数据对齐与打包、Table 8）、C（定性结果与图注），逐项核对实验口径和开放状态。
- **通读：** 已完成

## 定位与采集路径（§1、§3、Table 1）

解决问题：EGO 人手操作视频采集受佩戴、场地和人工成本限制；EXO 第三人称视频更丰富，但**直接转换第一视角是欠定的**，手部遮挡、观察角度、目标演员身份和跨镜头切换都会造成歧义。

1. 精确配对：两部带 LiDAR 的 iPhone Pro，同步采 EXO/EGO RGB、ARKit 位姿/时间、近场深度；先在“common-start”共同可见区域完成跨视图接续，再用 Pi3/DA3 等联合重建与 ARKit 轨迹进行 Sim(3) 度量对齐。
2. 单视角众包：只采真实 EGO（直接用作操作数据）或只采 EXO；互联网 EXO 同样能经过片段筛选与演员关联后进入生成管线。
3. 数据治理：保留原视频区间、视角身份（真实/生成）、相机标定、模型与 prompt 版本；派生视频跟原录制一起划训练/验证分组，避免泄漏。附录 B 给出最近帧对齐 20 ms 接受窗口及 30 fps 数据/24 fps H3 播放约定。

## 模型机制（§4、Fig.2）

- **输入条件：** EXO 视频 X + Qwen3.8-VL 从视频和结构化演员/手/物证据生成的操作文本 c + 可选指定演员的 EXO 截图 Ia。输出是 EGO RGB 视频，不是控制器动作或带真实关节力矩的轨迹。
- **生成器：** 以 MiniMax-H3 Ref2VA 为基座，冻结视频 VAE、H3 主干并微调 RGB LoRA；在视频 latent 上优化 flow-matching RGB loss。
- **训练期几何：** 把目标视角的 Plücker rays 注入 DiT EGO token（零初始化投影）；前 70% 训练以 0.5 概率注入，70–80% 线性衰减，末尾 20% 关闭。**推理仅需 EXO、文本和可选演员参考，无需目标相机标定/射线**。
- **多演员与跨镜头：** 结构化 hand-object 关系、演员参考图明确目标身份；多镜头训练把同步 EXO 视角在时间轴上切换、目标连续 EGO 不切换，不能外推到任意时间跳切或演员更换。

## 手–物交互监督（§5）

- **语义掩码分支：** RGB 和 mask 共用冻结 H3 主干、使用独立 LoRA；区分背景、操作物体、穿戴者手/前臂/衣袖。先 warm-up 掩码，后以 RGB loss + 0.1 × mask loss 联合训练；掩码梯度反传到 RGB 适配器。
- **ACE 手监督：** 冻结 ACE-Ego-Hand 提供伪标签/中间特征；HOT3D、ARCTIC、Re:InterHand 带标注样本校准读出器。ACEaux 在预测的 clean latent 上对齐特征、关节热图、二维关键点和手存在；MultiLayer 使用 DiT 6/12/18/25 层特征融合读出。
- **推理边界：** 手辅助读出器可移除；语义联合模型可另输出 mask。二者不是直接生成可执行机器人动作。

## 实验与关键数字（§6、Table 3–7）

- **验证集：** 546 clips / 86 captures；seen 475、unseen 71。448×448、30 fps；Table 3 的 141 帧对照为 545 clips（一个 EgoX/H3 几何先验失败被两边排除）。
- **49 帧（Table 3，All 546）：** EgoX → VBot：PSNR 12.630→15.856 dB；SSIM 0.4087→0.4894；LPIPS 0.5942→0.4955；CLIP-I 0.8427→0.8708；FVD 306.90→191.55。
- **141 帧（Table 3，All 545）：** EgoX/H3 → VBot：PSNR 12.579→15.826 dB；FVD 284.49→171.21。两基线的输入长度、几何先验及训练配方不同，属于**完整系统对比**，不是单模块严格控制消融。
- **语义监督（Table 4）：** PCK@5% 11.519%→16.538%；官方对象 IoU 5.471%→7.770%；但 joint coverage 92.433%→90.805%，不能说所有手部指标同时上升。
- **手监督（Table 6，Total 546）：** ACEaux 的 PCK@5% 11.52%→20.02%，EPE 57.09→47.42 px，手覆盖率 92.43%→95.18%；MultiLayer 的聚合图像指标更好，但手覆盖率 89.72%，低于 RGB-only。
- **随机相机条件（Table 7）：** updates 3001–5000 的训练 flow-MSE 0.13638→0.12557（-7.93%）；附录 A.2 指出历史基线的**数据池及学习率日程不同**，此比较仅具描述性，不能认定严格控制消融证明该机制收益。
- **数据计数（附录 A）：** 210,261 是 RGB 配对缓存条目配置数，**不等于**不同 EGO target 数或实际参与 checkpoint 训练的全部条目；mask 子集 13,188 个不同 EGO 时间窗口、49,012 个候选 EXO 配对。

## 局限与复现边界（§7、附录 A）

- 严重遮挡时多个第一视角结果都可能合理，物体不可见面、手指细节、头部运动无法仅从 EXO 可靠恢复。
- 伪标签与 ACE/SAM 读出器有偏差；官方 IoU unseen-object 仅 10 clips、4 captures，附录明确未建立该项收益。
- 多镜头只覆盖同步、同一连续操作；任意剪辑、跨时间跳转/演员更换仍需更丰富训练。
- 尚未公开可核实的运行入口；无真实机器人策略学习收益、动作标注可用率或闭环成功率实验，不能把视频质量提升当成 BC/VLA 真机性能提升。

## 对知识库的映射

- 主论文实体：[VBot Exo2Ego Policy](../../wiki/entities/paper-vbot-exo2ego-policy.md)
- 路线：[具身数据 Stage 1 采集 / Stage 4 扩增与合成](../../roadmap/depth-embodied-data.md)
- 关联：[Ego 采集](../../wiki/overview/ego-category-01-data-collection.md)、[EgoWorld](../../wiki/entities/paper-egoworld.md)

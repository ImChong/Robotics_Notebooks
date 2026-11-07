# Möbius: A Unified Motion-Language Model Bridging Understanding and Generation

- **论文：** https://arxiv.org/abs/2611.05279v1
- **版本：** 2026-11-03，v1（arXiv 头部标注 cs.CV）
- **作者：** Jiazhi Shu, Chengfeng Zhao, Jinyi Wang, Yubo Zhao, Zhiyang Dou, Ronghui Li, Yu Lu, Xinwei Jiang, Zijiao Zeng, Yuan Liu 等
- **机构：** 香港科技大学、腾讯游戏 Motus、麻省理工学院
- **项目：** https://jiazhishu.github.io/mobius/
- **官方仓库：** https://github.com/JiazhiShu/mobius-code
- **核对范围：** 通读 arXiv HTML（v1）全部正文、§1-§5、附录 A（数据、连续动作接口、任务构造、推理）、附录 B（评测协议与指标）、附录 C（补充结果）与表 1-4、S1-S9；参考文献仅核对正文引用，未逐条核查被引论文。未访问项目页与官方仓库的最新内容（见对应归档）。
- **通读：** 已完成

## 已核实摘录

### 1) 问题与方法

- 现有统一动作模型中，语言模型产生共享表示，去噪过程由外部扩散模块完成（§1，图 2）。Möbius 将干净动作、带噪动作与文本放入同一共享注意力，去噪在统一骨干中进行（§1、§3）。
- 表示：201 维 HY-Motion 特征，30 FPS，每帧占一个 Transformer 位置；包括骨盆平移(3)、6D 根朝向(6)、局部关节旋转(126)、根相对关节位置(66)（附录 A.2）。不使用离散 tokenizer 或预训练动作 VAE（§3.1）。
- 路由：文本、边界 token 与干净动作走理解参数；带噪目标动作走生成参数；所有 token 在共享注意力中交互（§3.2，Mixture-of-Transformers）。动作 token 内部为双向注意力，文本保持因果注意力（§3.2）。
- 训练两阶段（§3.3）：Phase I 从 Qwen3 初始化，在动作描述（M2T）上训练理解参数与动作投影；Phase II 用理解参数初始化生成参数，冻结理解参数，训练去噪目标（式 3-4）。编辑与时序补全通过改变观测帧集合实现，无专门架构（§3.3、附录 A.3）。
- 推理：Euler 积分 50 步，CFG 3.0，EMA 权重（附录 A.4、表 S2、S4）。

### 2) 规模与数据（易混淆处已核对）

- 0.6B 变体：Qwen3-0.6B 骨干，总参数 1.047B，仅用 HumanML3D 动作-文本数据（另加纯文本交织）（表 S1、附录 A.1）。
- 1.7B 变体：Qwen3-1.7B 骨干，**总参数 3.148B**，额外使用约 1,700 小时内部动作语料（附录 A.1、表 S1）。摘要中的「1.7B」指骨干规模，并非总参数量。
- 编辑变体 †：Phase II 加入 MotionFix 数据（附录 A.1、表 3）。

### 3) 主要结果（数值均来自论文表格）

- **T2M（表 1，HumanML3D，MS-272 评测器）：** Möbius-0.6B R@1 0.8209 / R@3 0.9670 / FID 6.1291 / MMDist 13.9973；Möbius-1.7B R@1 0.7968 / FID 10.6642 / MMDist 14.7441。HY-Motion(1B，大规模数据) R@1 0.7899 / FID 13.5194。注意：1.7B 使用额外内部数据，与仅 HumanML3D 训练的方法不可直接比较。
- **表述核对：** 正文称 1.7B 在 R-Precision、FID、MMDist 上最优，但表 1 中 0.6B 的 R@1 与 FID 更好（0.8209 / 6.13 优于 1.7B 的 0.7968 / 10.66）。以表格为准。
- **M2T（表 2，TM2T 协议）：** Möbius-0.6B BLEU-4 0.2032 / CIDEr 0.5754 / R@1 0.6056；MotionGPT3 对应为 0.1605 / 0.4077 / 0.5333。
- **编辑（表 3，MotionFix，MotionLab 协议）：** Möbius-0.6B† AvgR 2.8026 / FID 0.0955；MotionLab AvgR 3.7500；未编辑源动作 AvgR 3.5100。
- **时序补全（表 S9，附录 C.4）：** 预测任务 MPJPE：0.6B 为 7.177，最佳基线 Kimodo 为 8.781；关键帧插值 FID：1.7B 为 0.286，最佳基线 0.358。作者称观测帧保持（[P]-MPJPE）结果因任务而异，并非全面领先。
- **消融（表 4，仅 T2M+M2T，Qwen3-0.6B，MS-272 特征）：** 帧级编码优于 VQ 与 VAE；角色路由（ID 10）优于模态路由，后者使理解崩溃（ID 4）；理解先行后迁移（ID 10）优于仅生成（ID 6）、联合优化（ID 7）与随机初始化（ID 9）。注意：表 4 的 ID 10 FID 为 4.8772，与主表 6.1291 不一致，论文未解释。
- **附录 C.3：** 纯文本运动常识问答（表 S7）为定性示例，无定量评测。

### 4) 作者自述局限（§5）

- 实验集中于受控的单人动作；未来工作包括多人交互、精细手指动作、异构身体结构重定向、真实机器人部署。
- 论文未报告真机或仿真控制结果。

### 5) 声明

- 附录末尾 AI 使用声明：生成式工具仅用于语言润色（原文）。

## 对 wiki 的映射

- [Möbius 项目实体](../../wiki/entities/paper-mobius-motion-language.md)

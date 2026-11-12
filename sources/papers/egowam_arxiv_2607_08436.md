# EgoWAM（arXiv:2607.08436）

> 来源归档（paper，全文通读）

- **标题：** EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2607.08436>（v1，2026-07-08，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2607.08436v1>
- **项目页：** <https://gatech-rl2.github.io/egowam.github.io/>（项目页摘录另见 [egowam.md](./egowam.md)）
- **作者：** Baoyu Li\*、Xinchen Yin\*、Mengying Lin、Yixin Zhang、Danfei Xu（Georgia Institute of Technology；\*equal contribution）
- **代码：** 论文正文与附录未给出代码仓库地址
- **入库日期：** 2026-11-12
- **一句话说明：** 固定 HPT 骨干、flow-matching 动作头与数据配方，只替换世界预测目标（Pixel VAE / DINO / 3D flow），在三项真实双臂任务上研究野外 egocentric 人数据的人到机迁移。
- **核对范围：** 读了 arXiv v1 PDF 全部 23 页的正文文字（摘要、引言、相关工作、第 3–6 节、局限、参考文献目录）与附录 A（A.1–A.3）、B（B.1–B.3）、C（C.1–C.3）、D（D.1–D.3）全部文字；图 3、图 8 看了图像并读出数值，其余图（1/2/4/5/6/7/9/11/12/13）主要依据图注与正文描述，未逐图核对像素；图 3 的柱状数值为读图所得，未见对应表格。参考文献只看条目，未读被引论文。
- **通读：** 已完成

## 核心摘录

1. **问题（Abstract / §1）：** BC 把可迁移内容（物体、场景、任务语义）与不可迁移因素（人体形态、头部运动、行为风格）纠缠；论文研究 WAM 的未来预测通道是否是更好的训练信号，以及什么世界表征最利于人到机迁移。作者假设有效世界目标应抽象外观（D1）、跨具身一致（D2）、分离相机运动与环境变化（D3）（§4）。
2. **动作通道对齐（§3.1）：** 14 维末端空间（每臂 6-DoF SE(3) 加 1 维夹爪）；人手姿态重新表达到时刻 t 的设备坐标系以消除头动；人窗口 T_H=1 s、机器人窗口 T_R=1.5 s；1%/99% 分位数归一化。BC 共训目标为两个具身的条件 flow-matching 损失之和（式 1）。作者称这是其系统能支持的最强动作对齐基线。
3. **架构（§4.1，附录 B.1）：** 基于 HPT；具身专用 stem（ResNet-18 或预训练 DINO 的 ego stem、机器人额外 wrist stem、proprio MLP）投到 256 维，共享 trunk（256 维、16 块、8 头），64 个动作 token 与 16 个 future token（WAM）；动作头为 6 块 CrossTransformer，100 个动作 token，Beta(1.5,1.0) 时间步，50 步采样。
4. **三种世界目标（§4.2，附录 B.2）：** Pixel VAE（Wan 视频 VAE 潜码，16×16×16，输入 128×128；Pixel 为 6 块 DiT 从零训练，Pixel-PT 用预训练 VACE-1.3B/Wan 1.3B 初始化）；DINO（冻结 DINOv2-B 的 patch 特征，16×16×768，去 CLS 与 register，RAE 的 DiTDH 宽头）；3D flow（Track4World 追踪，用 Aria VIO 位姿稳定到时刻 t 相机系，28×40=1120 个锚点，100 步视野，目标形状 100×1120×3；机器人位移阈值 2 mm，人 10 mm）。
5. **联合训练与推理（§4.3，附录 B.3）：** L = L_action(robot+human) + λ(L_world(robot)+L_world(human))，λ=1；batch 32 机器人 + 32 人；AdamW lr 1e-4；单卡 L40S 每任务每方法约两天（Pixel-PT 用 2 卡）；推理丢弃世界头，仅动作路径，RTX 4090 上 30 Hz。
6. **实验设置（§5.1，附录 C，表 3）：** ARX5 双臂 + Aria 眼镜 + 两个 RealSense D405，Quest 3 遥操作。任务：cup-on-saucer、fold-clothes、bag-grocery（来自 EgoVerse flagship 任务集）。每任务机器人示教 300/360/300 条（2.5/3.0/2.5 h）；域内人数据各 2 h（约 1:1）；EgoVerse-A 人数据 20.5/21/7 h（总体约 10:1）。评测每方法 ID 20 次、OOD 20 次（10 个未见物体 + 10 次新场景），共 1800 次真机 rollout；报告归一化子任务分与成功率，误差条为 95% 置信区间 [66]。
7. **主结果（§5.2 Q1/Q2，图 3）：** 自然人数据下 BC 常变差或不涨（图 3：cup 域内 BC 成功率 -10 点，fold 域内 BC -15 点）；WAM 在多数设置为正增益（BC 在 OOD 上约 -5 到 +5 点）。Pixel 迁移弱；DINO 在 OOD 物体与场景增益最大（图 3：cup OOD 成功率 +43，bag OOD +40，fold OOD 场景 +70，均为读图数）；3D flow 在域内空间泛化增益最大（图 6），并给出最高整体分数。摘要称 DINO 的 OOD 泛化最高 4×、3D flow 域内提升 20–30%（摘要原话，未见其计算口径的单独说明）。fold-clothes 域内除 BC 外各方法均已饱和到 100%（图 3）。
8. **表征（图 4）：** UMAP 显示 BC 把人与机器人（甚至人与人）嵌入分开，WAM 对齐到共享空间（cup-on-saucer）。
9. **未对齐人数据消融（§5.2 Q3，图 8，附录 C.2）：** 仅 bag-grocery 在动作级共训下受益（动作天然对齐）。反事实：人用机器人不可执行的方式抓取。图 8 读数：BC 机器人单独 40%、域内共训 45%、未对齐共训 20%；3D flow 机器人单独 60%、域内共训 80%、未对齐共训 75%。
10. **对齐人数据消融（附录 A.1，图 9）：** cup-on-saucer，人演示者刻意模仿机器人视角与抓法。BC 在对齐后超过机器人单独基线；Pixel 35%→65%，DINO 50%→70%；3D flow 两种情形均 85%。
11. **人数据模态消融（附录 A.2，图 11，表 1）：** 仅 3D flow（人批次无动作监督）在三个切分上都优于仅动作；OOD Scene 上仅动作 0%、仅 flow 10%；动作 + flow 在每个切分最好。3D flow 预测损失：人流 0.23→0.22，机器人流 0.20→0.19（加动作标签后）。
12. **Pixel-PT 失败分析（附录 A.3，图 12）：** 预训练视频先验让模型“幻想”袋子已打开，策略跳过开袋阶段；EgoVerse 共训缓解该幻觉。
13. **RoboTwin 仿真（附录 D，表 4）：** 机器人到机器人迁移，aloha-agilex 为主，另三种机器人共训；100 个 held-out seed。pick-diverse-bottles 跨具身：BC 6、Pixel 11、DINO 28、3D flow 16；stack-bowls-three（评测物体外观被意外改变，作者标 †）：DINO 8、3D flow 16、BC/Pixel 0；hanging-mug 全部 ≤1%。
14. **作者自述局限（§6）：** 增益停留在语境层面（无法从人数据学到新的动作技能，如从 T 恤折叠迁到短裤）；每任务一个策略，未做多任务共训；“最佳”世界表征仍是开放问题；作者称研究不是定论而是起点。

## 对 wiki 的映射

- [paper-egowam-egocentric-human-wam-co-training](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)

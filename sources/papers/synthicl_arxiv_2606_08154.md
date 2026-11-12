# SynthICL（arXiv:2606.08154）

> 来源归档（paper）

- **标题：** SynthICL: Scalable In-context Imitation Learning with Synthetic Data
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.08154>
- **PDF：** <https://arxiv.org/pdf/2606.08154>
- **项目页：** <https://synth-icl.github.io>
- **代码：** https://github.com/sNiper-Qian/synthicl （项目页所列链接；本环境未能访问仓库内容，未核对其完整度）
- **作者：** Cheng Qian, Ruomeng Fan, Yifei Ren, Yilong Wang, Edward Johns（Imperial College London, The Robot Learning Lab）
- **版本：** v1，2026-06-06（cs.RO）；正文使用 CoRL 投稿模板（致谢段仍为模板文字）
- **入库日期：** 2026-11-12
- **一句话说明：** 只用 RGB 合成数据（Isaac Sim 渲染的伪示范）训练上下文模仿学习策略，flow matching Transformer 加辅助下一子目标图像预测；真机 16 个未见任务、测试时 1 条示范。
- **核对范围：** arXiv v1 PDF 全文（正文 §1–5、Limitations、参考文献、附录 A–H 及 Table 1–5、Figure 4/5 的文字与标注数值）；Figure 1/2/3/6/7/8 的图像内容仅读图注与正文描述；Figure 4/5 的柱高与曲线按标注读取，Figure 5 无逐点数值；未读项目页视频、未运行代码、未审阅仓库。
- **通读：** 已完成

## 核心摘录

1. **问题（§1）：** 上下文模仿学习（ICIL）需要大量多样数据；已有方法要么依赖大规模真机数据（ICRT），要么依赖点云（Instant Policy，IP）而存在 sim-to-real 差距并需真机共训练。
2. **数据（§3.2、附录 A）：** 沿用 IP 的伪示范（pseudo-demonstration）思路，不做物理仿真，只让夹爪沿物体中心路点运动并渲染；两处改动：PyBullet 换 Isaac Sim 渲染，用 GraspNet 预计算有效抓取位姿替代随机抓取。物体来自 RoboTwin-OD 与 ShapeNet，纹理来自 DTD。
3. **模型（§3.3、附录 B/D）：** 冻结 DINOv3（ViT-S/16+）、分解的时空注意力上下文编码器（8 层）、交叉注意力状态-上下文编码器（7 层）、ACT 式 flow matching 动作解码器（2 层，10 步 Euler）、独立的子目标图像解码器（2 层，MSE 重建，权重 1.0）；不使用本体状态；输入是分割后的 RGB（真机用 SAM + CUTIE）；上下文只含图像，不含动作；最多 6 帧上下文。
4. **仿真（§4.2，Table 1）：** RLBench 12 个未见任务，每任务 100 次；IP 72.4%，ICRT 60.1%，无子目标 72.2%，完整 75.0%。训练 100K 伪示范（PyBullet）。
5. **真机（§4.3，Table 2）：** Franka Research 3，3 路 RGB，16 任务各 20 次；IP 72.5%，ICRT 45.0%，无子目标 65.3%，完整 79.1%；训练 3M 条 Isaac Sim 轨迹。
6. **数据管线消融（§4.4，Figure 4，真机）：** IP 管线 9.3%，加 Isaac Sim 渲染 11.7%，加 GraspNet 抓取 64.6%，两者都加 79.1%。
7. **数据规模（§4.5，Figure 5，仿真）：** 3K 到 100K 条，两者都随数据增长；小数据下无子目标变体更好。
8. **消融（§4.6、附录 H）：** 时空分解注意力 75.0% 对完整注意力 63.4%；独立子目标头 75.0% 对联合解码器 76.1%；LoRA 微调 DINOv3 掉到 20.7%；小模型 64.8%；去掉腕部相机 60.3%，去掉侧视相机 64.6%。
9. **自述局限（Conclusion、附录 F）：** 依赖分割 RGB；高精度或富接触任务（紧配合插入、对齐）仍弱；评测以短时域为主；相同几何不同颜色的物体仍会混淆（RLBench Stack Cups）；目标离外部相机远时放置失败；无碰撞避让监督。

## 核对中发现的小问题（读者可自查）

- Table 1 的 Ours 一列，12 个任务数值取平均为 75.8%，而表中平均与全文引用均为 75.0%（IP、ICRT、无子目标三列平均与表一致）。
- 正文 §4.3 称各任务详述见附录 E，附录 E 只描述了 12 个任务，Rubbish in Bin、Close Drawer、Cube in Shape Sorter、Pull Screwdriver out 四个未见描述。
- 附录 D 的 Table 4 列出 State dimension 7，而 §3.1 称不使用本体状态（未说明该项用途）。
- 附录 H Table 5c 的列名写 Avg. Success，实际数值是 Cube in Shape Sorter 真机 20 次中的成功数。

## 对 wiki 的映射

- [paper-synthicl](../../wiki/entities/paper-synthicl.md)

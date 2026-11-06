# VFlash 技术报告

- **标题：** VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding
- **论文：** <https://research.nvidia.com/labs/lpr/vflash/assets/paper/vflash.pdf?v=20261105-2>
- **项目页：** <https://research.nvidia.com/labs/lpr/vflash/>
- **机构：** 英伟达；约翰斯·霍普金斯大学、香港理工大学、普渡大学、香港大学（额外机构）
- **版本：** 20261105-2 PDF（暂无 arXiv）。
- **通读：** 已完成
- **核对范围：** 2026-11-06 下载 PDF 并用 pdftotext 通读正文（摘要、§1 引言、§2 方法、§3 实验含表 1–8 与图 3–5、§4 相关工作、§5 结论）及附录 A–F 的设置与延迟统计；图中的纯数值标注以正文和表格文字为准，未逐像素核对图。

## 核心摘录

1. §1：DFlash 式轻量块扩散草稿模型一次前向预测整段 token，但贪心验证只接受到第一个与目标不一致处，`τ_ver` 常远低于块长；VFlash 以“结构合法即直接接受”取代该验证（式 2），文本仍保留目标验证。
2. §2.2：图元由类型、维度、长度（tool call 声明）确定布局；边界对齐（一个块覆盖完整图元）+ 位置对齐（每个槽位角色已知）。块长示例：2D+3D 轨迹 54（10 点）、框 6、稠密 3D 155、文本 16。坐标用 1,001 个特殊 token（0–1,000），文本分词目标用符号加五位数字的定宽网格。
3. §2.3 训练：先以 next-token 监督适配 VLM，再冻结，训练五层非因果 drafter；标签来自标注坐标而非目标续写，图元位置权重均匀；文本短块共同训练。推理：16 位 DFlash 块跑到图元起始 token 被接受，再一次草拟整块；逐位置词表约束，不增加神经网络评估；框附加 LocateAnything 空间置信度检查，失败部分由受约束 AR 回退。
4. §3 表 2/3：2D 轨迹在 ShareRobot-Bench 与 VABench-V 上六项误差均低于 AR 目标（HD 降 7%，DFD/RMSE 降 3–6%）；框在 COCO/Dense200 上 AR 回退版平均 F1 与 AR 持平且快 2.6–2.8 倍，IoU 0.95 处 F1 较低；无回退时 Dense200 平均 F1 约降 20 点。
5. §3 表 4：InternA1、AgiBot、MolmoBot 三个目标变体四项指标全优；3D-only ADE 降 18–49%，2D+3D ADE 降 28–61%；AgiBot 5 cm 精度 50%→73%、40%→74%；InternA1 2D+3D ADE 0.153→0.062 m；AR 联合输出 2D+3D 时 3D ADE 比 3D-only 高 15–51%。
6. §3 表 5：PointMotionBench 上 DAVIS ADE/FDE 降 27%/19%，HOT3D FDE 降 8%，WorldTrack FDE 增 15%，PWT 低 7–12%；单次 H100 rollout 3.21/3.86/3.78 s 对 AR 79.8/90.6/99.6 s（24.8/23.5/26.4 倍，附录 F.3，论文称初步测量）。
7. §3 表 6/7：MolmoSpaces 四套件 WAM 基线 62.6% → 70.2%，高于 PrimeR0 9.9 点；Pick & Place、Close 增益 11.6、12.8 点，Pick 仅 2.5；AR 生成的 2D+3D 计划 61.0%，换成图像状态 63.7%，3D-only 计划 69.3%。LIBERO 家族增益随扰动加大：0.8（LIBERO）、1.9（-Plus）、6.8（-PRO）点。
8. §3 表 8：冻结 VLM + 小动作头，NAVSIM SFT 87.1 PDMS（ReCogDrive 86.8），Waymo 7.79 RFS；仅对动作头做 RL 后 90.7 PDMS、7.86 RFS。
9. §3 延迟/附录 E：批大小 1，Thor/Spark 上 MolmoSpaces 每步延迟降 5.1–5.2 倍（Thor 3552.7 → 703.3 ms），H100 3.6 倍；NAVSIM 3.4–3.6 倍与 2.6 倍；Waymo H100 4.5 倍。AR 解码占 Thor 每步 75–87%，VFlash 将其降 12–42 倍；AR 每 token 一次 4B 前向 35–39 ms（Thor）；DFlash 在 MolmoSpaces 上子任务文本每次迭代提交 5.5 token，轨迹块内仅 2.2。
10. §4/§5：与 Spec-VLA（每遍仅 2.10–2.94 token、仍逐 token 验证）和 LocateAnything（仅固定大小 2D 图元）区分；结论称 25 倍为单次 rollout 估计，未来工作为共享观测下生成多个互补图元。

## 对 wiki 的映射

- [VFlash](../../wiki/entities/paper-vflash.md)

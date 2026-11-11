# SurgTAP-3D（arXiv:2611.11697）

> 来源归档（paper）

- **标题：** SurgTAP-3D: A Large-Scale Cross-Surgery Video Dataset with Contact-Aware 3D Motion Tracking
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.11697>
- **PDF：** <https://arxiv.org/pdf/2611.11697v1>（v1，2026-11-08 提交，cs.CV，共 19 页）
- **项目页：** <https://wrld.github.io/surgtap-3d/>（项目页写明 Accepted to IEEE Transactions on Robotics）
- **机构：** 香港中文大学 T Stone 机器人研究所 / 香港物流机器人中心（通讯作者 Yun-Hui Liu）；合作方含利物浦大学、上海交通大学、香港科技大学、浙江大学
- **入库日期：** 2026-11-11
- **一句话说明：** 把公开手术视频（机器人 + 手动，共 4,549 段动作片段、519,642 帧）经单帧接触区域提示 + 基础模型 + 动态 BA + 全局 4D 优化，自动标成 1,376,987 条 2D/3D 点轨迹（伪尺度）的数据集与基准。

## 核对范围

- **核对范围：** 读了 arXiv v1 PDF 全文（正文 I–VII 节、全部表格 I–XI、公式 1–25、参考文献列表）；图只看了第 6 页 Table II 与第 7 页 Fig. 3 的渲染图（Fig. 3 占比为读图近似值），Fig. 4–10 仅读图注与正文对应段落；论文无附录 / 补充材料（PDF 末尾即参考文献）。项目页 HTML 静态读过（文字与链接，不含视频）。未读被引的 Endo3R、TAPNext 等原文。
- **通读：** 已完成

## 开源状态（步骤 2.5）

- 项目页有 Code / Dataset 按钮，但链接为占位（`href` 为空或 `#`），页面「Resources」写 “Dataset & code: Release links will be added when available.”（静态读取，2026-11-11）。论文正文未给代码或数据下载地址。结论：**数据与代码均未见公开链接**；wiki 页因此不写 `code:`。

## 核心摘录（均出自论文，括号为章节 / 表号）

1. **规模与构成**（摘要、Sec. III-A、Tab. II）：4,549 个动作片段、519,642 帧、1,376,987 条 2D/3D 轨迹；来源为 GraSP、SAR-RARP50（机器人，前列腺切除）与 SLAM、SurgicalActions160、AutoLaparo、HeiCo（直肠 / 结直肠切除两个子集）、CholecT50、MultiBypass140（手动）。片段 5–30 秒、20–300 帧（Sec. III-B）。读 Fig. 3(c)：手动约占片段 2/3，机器人约 1/3（图上近似值）。
2. **统一动作类别**（Sec. III-A3）：dissection、tissue retraction、needle grasping、vessel clipping、coagulation 五类；有标签的源数据集映射到最近类别，无标签的人工按器械运动与上下文指定。
3. **「接触感知」的定义**（Sec. I）：显式标出可见的工具—组织交互区域，并以其初始化轨迹；**不假设力传感或接触物理建模**。每段只在首帧人工画一个接触区域框 + 3–5 个正点 / 3–5 个负点给 SAM2 出活动器械掩码（Sec. III-C1），其余自动。
4. **标注流水线**（Sec. III-C）：TAPNext 做 2D 跟踪 → Endo3R 出深度 / 位姿 / 内参 → 外部深度模型 [82]（正文称 DepthAnythingV3，Tab. VI）经最小二乘尺度 + 偏移对齐 → 基于 DROID-SLAM 更新模块的动态 BA（式 3–5，器械掩码与非静态区域降权）→ 全局 4D 精修（式 8–19：光流一致、表面约束 Chamfer、关节式 ARAP、时序加速度平滑、2D 轨迹正则；Adam，学习率 0.05，λ = 0.1 / 0.1 / 10 / 100 / 0.001）。
5. **标注质量指标**（Sec. III-D5、Tab. III）：SR 88.41、IMC 95.98、TCR 98.33、TSR 94.08（%，五类平均）。论文自己说明这些是「内部一致性指标，不是几何精度的直接度量」，且受跟踪器自身可见性 / 置信度、阈值（τc = 0.5，τs = 5 像素）影响。
6. **基准**（Sec. IV-A、Tab. IV）：在 SurgPose（器械）、SurgT、STIR（组织）上与 Type I/II/III 三类方法比较；3D 评测用**双目导出的参考轨迹**（式 24），论文承认它不是完美物理真值。平均：2D AJ / APD / OA 为 53.13 / 58.79 / 97.48，3D AJ / APD / OA 为 56.57 / 64.77 / 97.47；次优的 TAPNext 分别为 2D AJ 51.41、3D AJ 55.69 / APD 62.88。
7. **深度与位姿**（Tab. V、VI、VII）：SCARED 上 Abs Rel 0.124 → 0.103（Endo3R → 全流程），Hamlyn 0.170 → 0.156；SCARED 位姿 ATE 0.103、RPEr 0.195、RPEt 0.217。组件消融（Tab. VII）：深度精修只改深度（Abs Rel 0.124 → 0.115，位姿不变）；动态 BA 使位姿改善（ATE 0.112 → 0.103）且 Abs Rel 降到 0.108；全局 4D 精修再把 Abs Rel 降到 0.103，位姿指标不再变化。
8. **损失消融**（Tab. VIII）：去掉 Lreg 退化最大（AJ3D 56.57 → 55.41），去掉 Lflow 次之并使 Abs Rel 变差（0.103 → 0.108）。
9. **下游应用**（Sec. V）：（a）50 段自采 dVRK 单目视频 + 运动学，经 Sim(3)（Umeyama）对齐后算器械位移，ATE 0.0022 m、RPEt 0.0005 m（Tab. IX；无相机—机器人外参标定）；（b）10 个场景喂给 Endo-4DGS，替换 COLMAP 初始化后 PSNR 28.16 vs 27.82，LPIPS 0.1310 vs 0.2161（Tab. X）；（c）在本数据训练划分上微调 CoTrackerv3 与 DELTAv2，SurgT / SurgPose 的 AJ2D 提升约 0.8–1.6 点（Tab. XI）。
10. **论文自述局限**（Sec. III-D6、VI）：伪标注可能含噪；几何为序列内一致但**伪尺度**，用于相对 3D 运动分析，度量级机器人工作空间推理需额外标定；长期器械消失（> 10 秒）、大遮挡、快速运动易漂移或不稳定（Fig. 8）。

## 未在论文中找到的内容

- 数据集的许可、下载方式、训练 / 测试划分的具体切法（Sec. V-C 只说「our dataset 的 train split」，未给划分规模）。
- 对伪标注本身的人工逐点精度核查或临床专家评估。
- 与 SurgT / STIR 以外的跨数据集泛化实验（微调实验只在 SurgT 与 SurgPose 上评测）。

## 对 wiki 的映射

- [paper-surgtap-3d](../../wiki/entities/paper-surgtap-3d.md)

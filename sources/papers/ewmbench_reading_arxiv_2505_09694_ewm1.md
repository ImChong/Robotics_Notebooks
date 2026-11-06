# EWMBench 论文通读（具身世界模型生成评测）

> 来源归档（原文通读 · 支撑 `wiki/entities/ewmbench.md`）

- **标题：** EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models
- **作者：** Yue Hu, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, Guanghui Ren（AgiBot / SJTU / MMLab-CUHK / HIT）
- **类型：** paper / benchmark / world-model-evaluation
- **原文：** <https://arxiv.org/abs/2505.09694>（2026-11 通过 arXiv PDF v2 获取全文，pdftotext 提取）
- **一句话说明：** 面向具身世界模型（EWM）的视频生成基准：统一初始化（至多 4 帧 + 指令 + 可选 6D 轨迹），从场景一致性、末端运动、语义对齐与多样性三轴评测，数据取自 Agibot-World。
- **核对范围：** 通读正文 §1–§5（含 Table 1 对比、Table 2 结果、§4.2 人工评测、§4.3 轨迹指标互补性实验、§5 局限）及附录 A.1–A.4（预处理、YOLO-World 微调、任务与体素贪心采样、DINOv2 微调、HSD/nDTW/DYN 公式、失败案例）；参考文献仅浏览条目，未逐篇阅读被引论文；GitHub 仓库 README 与 HF 数据卡不在本归档范围内（见 `sources/repos/ewmbench.md`）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§1）：** 通用视频基准关注视觉保真、语言对齐、人类偏好；EWM 需要本体运动连贯、动作可信。操纵场景中背景、物体配置、本体形态应基本静止，仅位姿与交互随指令演化。
- **任务形式化（§3.1，式 1）：** V = v_norm(g_world(f_proc(I, L, T)))；I 至多四帧，T 为可选 6D 位姿序列；评测由多模态 LLM（语义）、轨迹检测器（低层轨迹）和视觉基础模型（特征）组成。
- **数据集（§3.2、A.2）：** 取自 Agibot-World，10 个任务（Toaster、Pour Water、Place Cutlery、Restock Freezer、Produce Ice、Factory Packing、Brush Bottle、Heat Food、Hang Showerhead、Store in Drawer）；每任务均匀采 100 个实例，左右末端位置体素化，用 3D IoU 相似矩阵 + 贪心（每次选与已选集平均 IoU 最低者）选出 10 条代表轨迹；高层任务拆 4–10 个原子子动作并配步骤字幕；引言称 30 个候选样本，实验为 10 任务 × 10 个真值 episode。
- **预处理（A.1）：** 参考图缩放到 640×480；提示用对应子动作字幕，并追加 “Keep the first-person view of the robot unchanged. Keep the first frame of this video unchanged.”；生成视频统一缩放 640×480、重采样 30 FPS。
- **轨迹提取（A.1）：** 微调 yolov8s-worldv2（Agibot-World 1451 帧人工标注，Freezer Restocking 与 Factory Packing 留作验证，100 epoch，Recall 0.917 / Precision 1.0），BoT-SORT 做时序关联；取双手凸包跨度最大的一只作为主轨迹；注意这里是 2D 检测轨迹。
- **Scene（§3.3、A.3.1）：** 以 dinov2-vitb14-reg4 初始化，在 Agibot-World 上无监督微调 20000 迭代，patch 嵌入的帧间 / 相对首帧余弦相似度；VBench 背景一致性在相机漂移时仍给高分（0.88–0.91），该指标能检出。
- **Motion（§3.3、A.3.2）：** HSD = 1/对称 Hausdorff 距离；nDTW = 1/NDTW 距离；DYN 对速度 / 加速度分布求 Wasserstein 距离，乘幅度归一因子 VR / AR，权重 α=0.007、β=0.003；每任务生成 3 条取 Hausdorff 最优。
- **Semantics（§3.3、§3.4）：** 视频 MLLM 产生全局字幕（对原指令算 BLEU）、关键步骤描述（对 MLLM 生成的真值步骤算 CLIP 分）、逻辑错误惩罚（幻觉操作、空间不合理）；多样性 = 1 − CLIP 全局特征相似度。
- **实验设置（§4）：** 7 个模型：OpenSora 2.0、LTX、COSMOS-7B（开源），Kling-1.6、Hailuo I2V-01-live（商用），LTX_FT、EnerVerse_FT（领域适配）；仅测图文到视频，因无开源的动作条件视频模型；10 任务 × 10 episode × 3 次生成 × 7 模型 = 2100 视频。
- **结果（Table 2）：** 总分 EnerVerse_FT 4.7010 > LTX_FT 4.5493 > Kling 3.8698 > Hailuo 3.4125 > COSMOS 3.2872 > OpenSora 3.1392 > LTX 2.9676；领域适配模型明显领先，尤其运动与语义。SceneC 与运动互补：静止视频场景分可很高而无有效运动。
- **人工评测（§4.2）：** 4 个模型（LTX_FT、Kling、Hailuo、OpenSora）人工排序，EWMBench 排名比 VBench 更贴近人工。
- **指标互补性（§4.3）：** 序列反转只有 nDTW 明显下降；插入离群点 HSD 与 DYN 下降；重复帧 nDTW 上升而 DYN 下降。
- **失败模式（§4.4、A.4）：** 领域适配模型有空抓（gripper 未闭合）；LTX 场景突变、常静止；COSMOS 与 LTX 画出人手而非机械臂，COSMOS 视角漂移；OpenSora 机械臂抖动、常生成静态视频；Hailuo 画面偏卡通。
- **局限（§5）：** 仅末端轨迹（未来加整臂状态）；固定视角（未来动态相机）；仅操纵任务（未来导航与移动操作）。

## 对 wiki 的映射

- [`wiki/entities/ewmbench.md`](../../wiki/entities/ewmbench.md)

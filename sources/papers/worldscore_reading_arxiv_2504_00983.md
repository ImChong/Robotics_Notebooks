# WorldScore 论文通读（世界模型/世界生成评测层视角）

> 来源归档（原文通读 · 支撑 `wiki/queries/embodied-eval-benchmark-selection-loop.md`）

- **标题：** WorldScore: A Unified Evaluation Benchmark for World Generation
- **作者：** Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, Jiajun Wu（Stanford）
- **类型：** paper / benchmark / world-generation / evaluation
- **原文：** <https://arxiv.org/abs/2504.00983>（2026-11 通过 arXiv PDF v2 获取全文，pdftotext 通读）
- **一句话说明：** 把世界生成分解为"下一场景生成"序列，用统一的（当前场景、下一场景、布局）规格同时评测 3D / 4D / I2V / T2V 模型，指标聚合可控性、质量、动态三方面共 10 项。
- **核对范围：** 通读正文 §1–§5 与附录 A–E（模型表 S1、数据集统计 S4、各指标定义 C.1–C.9、人类偏好验证 D、分辨率鲁棒性 S7、附加可视化 E）；参考文献列表与附录插图细节未逐条核对。该论文评测的是通用视频 / 场景世界生成，不含机器人动作条件世界模型，对具身场景的外推属本页推断。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§1）：** VBench 等视频基准只评单场景质量、无布局控制，与需要相机输入的 3D/4D 方法不兼容；图 1 中 VBench 给 A/B 分别为 80.28 / 83.06 近似打平，而 WorldScore 为 70.70 / 46.28，能识别 B 不生成新场景、不遵循相机运动。
- **世界规格（§3.1）：** 三元组 (C, N, L)：当前场景（图像 + 文本）、下一场景文本、布局（相机轨迹 + 相机运动文本）；所有模型输出统一为视频再评测；静态世界测可控性与质量，动态世界固定相机测动态。
- **数据集（§3.2，表 S4）：** 3000 条：2000 静态（室内 5 类 + 室外 5 类，各 100 × 写实 / 风格化）+ 1000 动态（5 类运动：铰接 / 形变 / 流体 / 刚体 / 多运动）；7 种风格；8 种相机运动；小世界（1 个新场景）与大世界（3 个新场景，占静态 20%）。
- **10 项指标（§3.3，附录 C）：** 可控性 = 相机（旋转与平移误差几何平均，DROID-SLAM 估位姿）、物体（Grounding DINO 检测率）、内容对齐（CLIPScore）；质量 = 3D 一致性（重投影误差）、光度一致性（光流前后向 AEPE）、风格一致性（Gram 矩阵）、主观质量（CLIP-IQA+ 与 CLIP Aesthetic 均值，经 400 人 2AFC 研究选出，一致性 0.637，上界 0.772）；动态 = 运动准确率（区域内外光流差）、运动幅度（光流中位数）、运动平滑度（帧插值重建误差）。线性归一化到 0–100 的经验上下界；WorldScore-Static 为可控性 + 质量均值，Dynamic 再加三项动态；3D 模型动态项记 0。
- **验证（附录 D）：** 跨分数档的 2AFC 偏好一致性多数 > 70%（相机控制 60 比 30 档 71.2%）；同一视频缩放到 256×256 后各指标变化 ≤ 0.83，对分辨率与长宽比稳健。
- **结果（§4.1，表 2，图 7）：** 评 20 个模型（13 视频、6 个 3D、1 个 4D）。3D 场景生成模型静态最佳（WonderWorld 72.69、LucidDreamer 70.40），优于最好的视频模型 CogVideoX-I2V（62.15）；视频模型相机可控性弱（最好 CogVideoX-T2V 40.22，低于任何 3D/4D 模型）；长序列与室外场景尤弱；开源 CogVideoX-I2V 的 Static/Dynamic（62.15 / 59.12）超过闭源 Gen-3、Hailuo；运动幅度越大平滑度越低，且幅度与准确率弱相关；T2V 更易控制相机而 I2V 质量更高、倾向停留在输入视角。
- **局限（文中结论与附录）：** 度量大量依赖现成模型（SLAM、光流、检测器、CLIP），归一化用经验上下界；4D 仅评 4D-fy 一个模型且已降低迭代步数；用 z-score 把部分指标压到 25–75 区间以增强区分度。

## 对 wiki 的映射

- [`wiki/queries/embodied-eval-benchmark-selection-loop.md`](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)：支撑 ② 层（世界模型预测保真度评测）"视频逼真度单指标不足以区分世界生成能力、需拆可控性 / 质量 / 动态"的论据。
- 既有摘要归档：[worldscore_arxiv_2504_00983.md](worldscore_arxiv_2504_00983.md)

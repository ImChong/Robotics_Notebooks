# Thermo-VLA（arXiv:2611.11981）

> 来源归档（paper）

- **标题：** Thermo-VLA: A Thermal-Aware Vision-Language-Action Model with Gated Residual Modulation for Robotic Manipulation
- **类型：** paper
- **作者：** Jingxuan Chen, Junshi Luo, Chencheng Dong, Xiaojian Nie, Tongge Huang, Qiang Huang（PDF 首页未列机构）
- **arXiv：** <https://arxiv.org/abs/2611.11981>（v1，2026-11-09，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.11981v1>
- **项目页：** <https://j-x-chen.github.io/Thermo-VLA-web/>（含补充材料 PDF：`assets/appendix.pdf`，6 页）
- **入库日期：** 2026-11-11
- **一句话说明：** 在 π0.5 上加 ThermoRes Encoder：把热成像按背景校准拆成冷 / 热两个属性视图，由文本门控预测权重，以残差形式加到头部相机 RGB token 上；在 Unitree G1 双臂真机 6 个任务上评测，另把同一机制迁移到颜色属性。
- **核对范围：** 逐段通读 arXiv v1 PDF 全文（8 页，含正文、全部表格 I–V、参考文献）与项目页补充材料 PDF（6 页，含符号表、硬件、训练细节、数据集表 S2、实现配置表 S3、RGB–热成像对齐、低秩分析与表 S4）；项目页 HTML 文字已读。图只看了 Fig. 2 版面与各图题注，Fig. 3/4、补充图 S1–S5 的图像内容未逐幅检查；项目页演示视频页面显示 “No demonstration videos found”，未核对。未运行任何代码（作者未发布代码）。
- **通读：** 已完成

## 开源状态

- 项目页（2026-11-11 访问）的 Code、Data 均标 **Coming Soon**，论文 PDF 内未给仓库链接 → 判为 **未开源**（代码与数据均未公开）。
- 项目页页脚写 “This work is currently under review”。

## 核心摘录

1. **方法（§III）：** 基线为 π0.5（Gemma 骨干 2B + 动作专家 300M，表 S3）。ThermoRes Encoder = 双分支属性分解 + 门控残差注入。热成像先取上边界多个 patch 的均值之众数作背景参考 b，校准到固定灰度 µ（µ=80，表 S3），再按式 (2) 拆成 x_cold、x_hot；两分支共享 SigLIP 系热成像编码器 E_t 与 RGB 投影器 P_r。
2. **门控（式 3-4）：** 文本 token 作 query，冷 / 热 token 作 key-value 做共享参数的交叉注意力；拼接 [文本; O_c; O_h; O_c−文本; O_h−文本] 经 LN + MLP（10240→512→256→2）后 softmax 得 [α, β]（二者和为 1）；头部 RGB token 分别加 αZ_cold 与 βZ_hot，两份与左右腕 RGB、文本 token 组成前缀 Z_T，动作生成沿用 π0.5 的 flow matching（动作块 H=50，10 步积分）。
3. **平台与数据（§IV-A、表 I、附录 II/IV）：** Unitree G1 EDU + Dex3-1 手，头部 D435i RGB + Tiny1-C 热像（256×192），两腕 D405；PICO XR 遥操作，共 1,307 条专家示范（156/303/236/340/173/99），扩展 LeRobot v3 格式。
4. **主结果（表 II）：** 每条指令测 20 次。四个温度相关任务（II/IV/V/VI）Thermo-VLA 为 1.00/0.86/0.58/1.00，均值 0.86；π0.5(RGBT) 为 0.80/0.85/0.50/0.55；ThermoAct 为 0.45/0.55/0.00/0.50。RGB-only 的 I、III 任务：Thermo-VLA 0.88/0.76，原版 π0.5 0.93/0.84，π0.5(RGBT) 0.85/0.71。
5. **泛化（表 III）：** 未见物体 0.79、光照变化 0.76、未见桌布 0.79（π0.5(RGBT) 为 0.64/0.68/0.72）；Task IV 温度属性重赋予 0.60（π0.5(RGBT) 0.10）。
6. **消融（表 IV）：** 去掉门控残差注入后 Task VI（更小的未见目标）0.85→0.60；去掉双分支改用伪彩色输入后 Task II 光照变化 0.90→0.85、Task IV 桌布 0.88→0.80；热残差置零后温度相关任务约 0.5。
7. **颜色迁移（§V、表 V）：** RGBRes 在 LIBERO（Franka，Task VII）与真机杯子选择（Task VIII）上平均 0.572→0.775；残差 PCA（表 S4）：热成像任务 top-32 成分保留 83.1%/84.4%/89.6% 能量，颜色任务 top-8 保留约 94–96%。
8. **论文自述局限（§VI）：** 计划在更大规模热成像数据上预训练热成像编码器以更好保留 RGB-only 能力；在冷 / 热分解之外增加温度幅值信息，以区分 Task V 中的 cold 与 cool。

## 对 wiki 的映射

- [paper-thermo-vla](../../wiki/entities/paper-thermo-vla.md)

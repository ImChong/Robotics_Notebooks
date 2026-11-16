# RCT: A Robot-Collected Touch-Vision-Language Dataset for Tactile Generalization（原文与仓库核对笔记）

- **来源：** <https://arxiv.org/abs/2606.31694>（`arXiv:2606.31694v1`，2026-06-30）
- **项目页：** <https://faerber-lab.github.io/RCT/>
- **代码仓库：** <https://github.com/faerber-lab/RCT>（Apache-2.0；作为上游 [TVL](https://github.com/Max-Fu/tvl) `f489503` 的补丁叠层）
- **数据集：** <https://figshare.com/s/a5ed417ba6602ccad0f6>（CC BY 4.0）
- **作者与机构：** Jingbo He, Michael Färber（ScaDS.AI, TU Dresden）, Roberto Calandra（LASR Lab, TU Dresden）
- **核对日期：** 2026-11-14
- **核对范围：** 论文全文 `arXiv:2606.31694v1`（第 1–985 行，含摘要、方法、4 类实验与附录）及 `faerber-lab/RCT` 仓库 `README.md`、`CHANGES.md`、`tvl_modifications.patch`、`tools/splits/*`、`tools/leakage/*`、`experiments/examples/*`
- **通读：** 已完成

---

## 开源与数据核查

- **开源状态：** **已开源**。代码以 Apache-2.0 许可发布（通过 `tvl_modifications.patch` 叠加在 TVL commit `f489503` 之上，辅以 `tools/` 脚本与 `experiments/experiments.yaml`）；数据本体托管于 Figshare，采用 CC BY 4.0 许可。

---

## 原文核心事实摘录

1. **采集硬件与协议：** 机械臂末端搭载旋转转接件，安装 **3 颗 DIGIT** 视觉触觉传感器（`d1`、`d2`、`d3`），对 **122 种工业参考材料**（来自 Modulor GmbH *Musterkiste*，覆盖 7 个大类：Plastic/Rubber 33、Paper/Cardboard 45、Metal 20、Textiles/Leather 10、Wood/Bamboo/Cork 7、Crafts 4、Small Items 3）进行垂直按压采集。
2. **数据规模与结构：** 共 **29,279 帧** DIGIT 触觉图像，组织为 **1,832 条按接触顺序排列的接触序列（contact sequences）**。每次按压从刚接触开始以 `0.10 mm` 步长下压（通常每条序列 15–17 帧），每帧同步记录实时接触力（per-frame contact force）。位置 p1–p4 密集采集（各约 5,800 帧），p5 覆盖 75/122 种材料（约 3,500 帧），p6–p18 稀疏分布（最低约 50 帧）。
3. **三模态设计：**
   - **Touch：** DIGIT RGB 触觉帧 + 每序列首帧背景参考帧；
   - **Vision：** 每种材料 1 张 `2048×1536` 高分辨率工作室参考照片（注：由所有同材料触觉帧共享，故触觉→视觉检索采用多正样本 `V @ V.T` 评估）；
   - **Language：** 基于 54 词可控触觉词表（物理/纹理/材料/结构/形状 5 组），由 2 名人类标注者通过实际触摸分别选出 3–5 词（默认主标注取人类标注并集 union）。人类标注者间平均 Jaccard 为 `0.294`，而 Gemma-3-27B 仅凭外观照片生成的标注与两位人类标注者的 Jaccard 仅为 `0.110` / `0.108`（且 38% 材料与人类零交集）。
4. **切分轴与模型配方：** 支持跨材料（holdout K=2/5/20）、跨类别、跨接触序列、跨接触位置、跨传感器（如留出 `d3`）以及帧级随机对照切分。基线沿用 TVL ViT-Base 配方（`vit_base_patch16_224` + OpenCLIP ViT-L/14，batch 256，200 epochs，warmup 10，lr `1.5e-4`，wd `0.05`，余弦退火，InfoNCE；tactile→text 软标签阈值 `0.6356`）。
5. **关键量化结果：**
   - 帧级随机切分触觉→文本 R@1 为 `80.01%` → 切换到**序列不重叠**测试池降至 `62.33%`（**-17.7 pp** 轨迹泄漏虚高；直接序列留出训练为 `62.44%`） → 进一步切换到**未见材料 K=20**（seed 42 diverse）降至 `20.35%`（再降 **-42.0 pp**；3 种子均值 `25.13 ± 6.14%`，类别均衡切分 `29.79 ± 2.93%`）。
   - 对已发布 TVL/HCT 切分的泄漏审计：HCT 测试集中 **279/279 (100%)** 接触序列同时出现在训练集（同 `<idx>-<timestamp>` 目录，中位仅隔 1 步），免训练原始像素 L2 最近邻在 **350/356 (98.3%)** 测试样本上直接召回正确所属序列（跨序列/同序列 L2 距离中位比 `2.6×`）。
   - 帧密度抽样实验（序列留出池）：全量 `Full`（22,576 帧）`62.44%` tac→text / `70.95%` tac→vis；仅取最深 5 帧 `Deep5`（7,055 帧）`59.52%` / `65.96%`；跨深度均匀采样 5 帧 `Uniform5`（7,055 帧，约 1/3 数据量）反而升至 `68.18%` / `78.29%`。
   - 跨传感器留出（`d3` 未知、`d1+d2` 训练）：`21.51%` tac→text / `13.00%` tac→vis，难度接近跨材料；跨位置留出（`p5`）：`65.46%` / `74.16%`。
   - 下游线性探针（K=20 冻结编码器）：类间-类内余弦间隔 `m = d_inter - d_intra` 上 RCT `0.297 ± 0.020` 显著高于 TVL `0.048 ± 0.010` 与随机基线 `0.004 ± 0.001`；7 类材质分类准确率 RCT `49.8%` vs TVL `34.2%` vs 随机 `37.8%`；但二分类硬/软仅 `52.2%`（接近 59% 多数类基线），表明视觉参考仍主导可学特征。

---

## 仓库代码核对

- `tvl_modifications.patch`（+190 / −19 行，修改 TVL 5 个文件）：
  - `tvl_enc/tacvis.py` 新增 `RCTDataset(TacVisDatasetV2)`，仅依赖 CSV（`vision, tactile, background, caption`）驱动切分并复用 v1 中心裁剪；
  - `tvl_enc/main_pretrain.py` 注册 `--datasets rct` 及 `--rct_train_csv` / `--rct_test_csv`；
  - `tvl_enc/loss.py` 增加 `use_vision_gt_distribution`（计算 `V @ V.T` 多正样本标签）；
  - `tvl_enc/tools/visualize_affinity.py` 增加 `--multi_positive_vision` 并改写为 256 样本分块推理以防大规模测试池显存溢出；
  - `tvl_enc/util/misc.py` 兼容 `torch.load(..., weights_only=False)`。
- `tools/splits/`：`make_material_holdout_split.py`、`make_category_holdout_split.py`、`make_trajectory_split.py`、`make_axis_holdout_split.py`、`make_density_filter.py` 等生成全部切分。
- `tools/leakage/`：`audit_tvl_trajleakage.py`（检查轨迹目录与步距重叠）与 `audit_hct_raw_pixel.py`（像素 L2 泄漏验证）。

---

## 映射到 Wiki

- [dataset-rct-robotic-contact-tactile.md](../../wiki/entities/dataset-rct-robotic-contact-tactile.md)

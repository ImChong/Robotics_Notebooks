# PoseScript 原始论文与官方仓库精读档案（arXiv:2210.11795v3 / ECCV 2022 & TPAMI 2024）

- **标题：** *PoseScript: Linking 3D Human Poses and Natural Language*（ECCV 2022 初版扩展为 IEEE TPAMI 2024 版，DOI `10.1109/TPAMI.2024.3407570`）
- **作者与机构：** Ginger Delmas（IRI CSIC-UPC / NAVER LABS Europe）、Philippe Weinzaepfel、Thomas Lucas、Grégory Rogez（NAVER LABS Europe）、Francesc Moreno-Noguer（IRI CSIC-UPC）
- **原文链接：** <https://arxiv.org/abs/2210.11795> · <https://europe.naverlabs.com/research/computer-vision/posescript/>
- **官方仓库：** <https://github.com/naver/posescript>
- **全文核对：** 已通读 `arXiv:2210.11795v3` 原文抽取文本（全文 3,629 行）
- **通读：** 已完成
- **核对范围：** 全文 I–X 节与 Appendix A–C（姿态采样策略、AMT 两步歧视性人工采集、87 类 posecodes/super-posecodes 与“涟漪消冗”规则、Text-to-Pose 检索、条件 VAE 姿态生成、自回归姿态描述生成及与 CLIP 对比消融），以及官方仓库 `README.md`、`LICENSE`、`src/text2pose/config.py`、`posescript/{README.md,compute_coords.py,posecodes.py,captioning.py}` 与 `{retrieval,generative,generative_caption}/script_*.sh`
- **关联维基：** [PoseScript 实体页](../../wiki/entities/dataset-bfm-posescript.md)

## 核心数据与仓库核对要点

1. **数据集版本、规模与许可：**
   - **PoseScript-H（人工标注）：** 论文 `v3`（TPAMI 2024）统计共 **6,283** 条 AMT 人工描述（仓库 `v2` 发布文件为 `posescript_human_6293.json`；ECCV 2022 初版为约 3.9K/6K 阶段），平均每条 `54.2 tokens`（`50.3` 词）、词表 `1,866` 词，平均覆盖 `6.2` 个身体部位，其中 `86%` 使用以人身为中心的自我参照关系、`41%` 存在隐式左右侧推演、`24%` 使用类比（如“像躲子弹/热身”）；
   - **PoseScript-A（自动规则合成）：** 从 `14,096` 条 AMASS `SMPL+H G` 序列剔除首尾 25 帧 T-pose 后按 25 帧间隔叠加最远点采样（MPJE）选出 **100,000** 帧多样化 3D 姿态（`DanceDB 28%`、`CMU 20%`、`BioMotionLab 13%` 占比最高），每帧随机合成 **3** 条自动描述（`posescript_auto_100k.json`，共 300K 条文本，生成 100k×3 仅耗时不到 10 分钟）；
   - **划分与许可：** 按源 AMASS 序列划分约 `70% train / 10% val / 20% test`（无同序列跨划分泄露；`PoseScript-H` 测试集 `1,234` 样本，`PoseScript-A` 测试集 `19,990` 样本）。数据集与 `text2pose` 代码统一采用 **CC BY-NC-SA 4.0** 非商业许可（版权归 CSIC-UPC 与 Naver Corporation），依赖的 AMASS、BABEL、SMPL-H 与 VPoser 亦各自适用原学术非商业授权。

2. **规则化自动描述流水线（§III-C & `src/text2pose/posescript/`）：**
   - `compute_coords.py` 使用 `human_body_prior.BodyModel` 从中性标准体型（`n_betas = 16`、`SMPL-H 52 关节`）导出关节 3D 坐标并绕 X 轴旋转 `-90°` 对齐前视朝向，外加掌心与躯干中心虚拟关键点；
   - `posecodes.py` + `captioning.py` 评估 **77 个基础 posecode**（4 角度 + 22 欧氏距离 + 34 X/Y/Z 轴相对位置 + 13 俯仰翻滚 + 4 触地距离）与 **10 个二值超级 posecode**（如 `kneeling`、`kneel on left/right`、`torso horizontal`），分箱前对连续值叠加随机主观容差噪声 `[-η_τ, +η_τ]`；
   - 经统计频度过滤（`>=60%` 平凡类别忽略、`<6%` 不可跳过、其余以 `15%` 概率随机跳过）与两类涟漪消冗（传递关系 `a < b < c` 删 `a < c`，以及频繁共现规则 `X => Y`），以 `95%` 概率触发 4 种句法聚合，并可选附加 BABEL / DanceDB 动作标签句（覆盖 64,758 个姿态）。

3. **三大下游基准与可运行脚本核对：**
   - **跨模态检索（`retrieval/script_retrieval.sh` / `train_retrieval.py`）：** 基于 VPoser 编码 `(52, 3)` 轴角姿态与文本编码器（GloVe-BiGRU 或冻结 DistilBERT + 4 层 Transformer）对齐 512 维联合空间，优化对称/标准 `BBC` 对比损失；预训练 + 微调使 `PoseScript-H` 的 `mRecall` 从从头训练的 `23.0%` 提升至 `40.9%`，叠加 Transformer 与左右镜像增广达 **`45.3%`**（同条件下通用 `CLIP` 渲染图文检索仅为 `1.6%`）；
   - **文本条件姿态生成（`generative/script_generative.sh` / `train_generative.py`）：** `CondTextPoser` 条件 VAE 将 `32` 维潜空间按文本先验分布对齐，解码输出 6D 连续关节旋转，A→H 预训练微调将 `PoseScript-H` 上生成 `FID` 从 `0.29` 压低至 `0.04`（Transformer + 镜像达 **`0.03`**）；
   - **姿态描述生成（`generative_caption/script_generative_caption.sh` / `train_generative_caption.py`）：** 基于 Cross-Attention 的 4 层 8 头 Transformer 解码器自回归生成姿态描摹文本，A→H 预训练 + 镜像增强将检索重建 `R@1` 从 `24.39%` 提升至 **`88.98%`**（`BLEU-4 = 13.22`、`ROUGE-L = 34.07`、`METEOR = 33.09`）。

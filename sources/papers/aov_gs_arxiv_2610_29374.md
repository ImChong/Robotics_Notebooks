# AOV-GS（arXiv:2610.29374）

> 来源归档（paper）

- **标题：** AOV-GS: Active Open-Vocabulary Semantic Mapping with 3D Gaussian Splatting
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.29374>
- **PDF：** <https://arxiv.org/pdf/2610.29374>
- **代码：** <https://github.com/OptimusFaber/AOV-GS>
- **入库日期：** 2026-10-29
- **核对版本：** arXiv:2610.29374v1（2026-10-21）
- **原文 HTML：** <https://arxiv.org/html/2610.29374>
- **核对范围：** 已阅读 arXiv v1 全文的摘要、引言、相关工作、§3.1–3.5（建图、探索得分、SAM–CLIP、语言场）、§4.1–4.6（协议、Replica、Matterport3D、重建、成本与消融）、结论与论文参考列表；另核查官方仓库 README、`scripts/aov-gs/README.md` 以及三个阶段的 shell 入口。
- **通读：** 已完成
- **复现状态：** 未实际执行官方训练和评测命令；数值仅引用论文原始表格。
- **线索来源：** [机器人缺的，是更多示范，还是更多成功解法？｜附代码与项目入口](../blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)（具身智能小站，2026-10-29）
- **一句话说明：** AOV-GS以冻结的SAM与CLIP为候选视角排序，再拟合开放词汇语言场；室内仿真显示语义地图收益，但实验使用真实相机位姿，尚未覆盖SLAM定位噪声。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-29）：仓库可达（git ls-remote）。

## 核心摘录（已核对原论文）

1. **双阶段探索：** Stage 0 先对渲染不可见区域与路程形成几何得分，再仅对 Top-K（默认 8）候选视角使用 SAM 分割与 CLIP 向量相对关键帧库衡量语义新颖度；Stage 1+ 恢复几何优先的规划。
2. **并行语义采集：** 已执行轨迹上的关键帧由冻结的 SAM 和 CLIP 抽取区域嵌入并进入语义库；只用于候选评分的合成视角向量不写入库。随后使用 LangSplatV2 在高斯场上离线蒸馏稀疏语言特征。
3. **评测条件：** Replica 八个室内场景和 Matterport3D 五个场景，Habitat 给出真值相机位姿作为 oracle SLAM 输入；不能据此证明真机 SLAM 漂移鲁棒性。
4. **开放词汇结果：** Replica 八场景语言场 mIoU 34.6%，比同 SAM–CLIP 配置的仅几何探索高 1.3 个百分点；Matterport3D 五场景 23.1%，高 1.8 个百分点（论文摘要及表 1/5）。加入 CorrCLIP 后 Replica 35.6%，高于同 CorrCLIP 基线 2.0 个百分点。
5. **成本与局限：** Stage 0 仅 Top-K 候选增加语义计算，论文 office0 的每个候选约 8 秒（RTX 4070 Ti SUPER）；语言场仍需离线拟合。论文部分场景出现退化，CorrCLIP 也不保证在所有场景提高分数；真实传感器噪声和追踪漂移未评估。

## 源码入口核对

- 端到端入口 `bash scripts/aov-gs/pipeline_gs_open_vocab.sh office0` 按序调用 `01_slam_exploration.sh`（`python src/main/activesgm.py --cfg ...`）、`02_validate_features_langsplatv2.sh`（仅检查 `language_features/`，V2 不再单独训练 AE）、`03_train_gaussian_lang_field_langsplatv2.sh`（`scripts/train_language_field.py --lang_mode langsplatv2`）。
- Stage 01 输出 `splatam/final/params0.npz` / `params.npz`、`keyframe_poses.json`、`language_features/`；Stage 03 输出 `lang_field.pt`。模型效果与运行环境仍未独立复验。


## 对 wiki 的映射

- [paper-aov-gs](../../wiki/entities/paper-aov-gs.md)
- [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md)

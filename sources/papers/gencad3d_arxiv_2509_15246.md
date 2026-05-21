# GenCAD-3D: CAD Program Generation using Multimodal Latent Space Alignment and Synthetic Dataset Balancing（arXiv:2509.15246）

> 来源归档（ingest）

- **标题：** GenCAD-3D: CAD Program Generation using Multimodal Latent Space Alignment and Synthetic Dataset Balancing
- **缩写：** **GenCAD-3D** / **GenCAD3D**
- **类型：** paper / 逆向工程 / 点云·网格→CAD / 对比学习 / 潜扩散 / 数据增强
- **arXiv：** <https://arxiv.org/abs/2509.15246>（HTML：<https://arxiv.org/html/2509.15246v1>）
- **期刊：** ASME Journal of Mechanical Design（JMD，DOI 入口见项目 BibTeX：<https://doi.org/10.1115/1.4069276>）
- **项目页：** <https://gencad3d.github.io/>
- **代码：** <https://github.com/yunomi-git/GenCAD-3D>
- **数据与权重：** Hugging Face [`yu-nomi/GenCAD_3D`](https://huggingface.co/datasets/yu-nomi/GenCAD_3D) · [`yu-nomi/GenCAD_3D`](https://huggingface.co/yu-nomi/GenCAD_3D)（weights）
- **作者：** Nomi Yu, Md Ferdous Alam, A John Hart, Faez Ahmed（MIT）
- **入库日期：** 2026-05-21
- **一句话说明：** 在 **GenCAD** 图像路线之上，用 **对比学习** 将 **点云 / 点云+法向 / 网格** 潜空间与 **冻结的 CAD 自编码器潜空间** 对齐，再以 **条件潜扩散** 做 **几何→CAD program** 的 **生成与检索**；**SynthBal** 针对 DeepCAD **序列长度长尾** 合成增广，显著降低 **无效 CAD** 比例并改善 **高复杂度** 重建。

## 摘要级要点

- **问题：** 从 **非参数化几何**（点云、网格）恢复 **可编辑 CAD program** 是逆向工程核心，但公开数据（如 DeepCAD 约 17.8 万程序）对 **长序列/高复杂度** 样本严重不足，平均指标易被简单形状主导。
- **框架（三阶段，Fig.3）：**
  1. **CAD 表示学习：** 因果 Transformer 自编码器（与 GenCAD 同族）将 sketch–extrude 命令序列编码为 \( \mathbf{z}_{\mathcal{C}} \)。
  2. **多模态对比：** 模态专用编码器（**DGCNN** 点云；**DGCNN+法向**；**FeaStNet** 网格）与冻结 CAD 编码器用 **对比损失** 对齐 \( \mathbf{z}_{\mathcal{M}} \) 与 \( \mathbf{z}_{\mathcal{C}} \)。
  3. **条件潜扩散：** prior \( p(\mathbf{z}_{\mathcal{C}} \mid \mathbf{z}_{\mathcal{M}}) \) + 已训练 decoder，从几何潜生成 CAD 潜再解码为命令序列。
- **SynthBal：** 针对 **命令序列长度** 不平衡，对欠表示长度 **合成 CAD program** 并合并划分；论文报告无效生成率由约 **3.44%** 降至 **0.845%**，高复杂度 **Chamfer** 中位误差最高降约 **89%**；另提供 **SynthBal_1M** 等子集供训练。
- **编码器对比：** 网格路线相对纯点云在命令/参数精度与大规模检索 top-1 上有一致提升（论文给出相对 **~15% / ~11%** 量级叙述，以正文表格为准）。
- **命令集（本研究范围）：** Line / Arc / Circle + Extrude + EOS；架构声明可扩展至 **revolve、fillet** 等工业命令，但实验聚焦 **sketch-and-extrude**（与 DeepCAD 编码兼容）。
- **真实扫描集：** 发布 **51** 件 **3D 打印 + 激光扫描** 部件及对应 CAD program（**GenCAD3D_Scans**，约 700 MB），强调物理扫描噪声与纯合成数据的差异。
- **下游工具链：** 推理可输出 **h5** 程序；仓库提供 **Onshape API** 脚本将程序推送到 **Part Studio**（需用户 API key）。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/gencad-3d.md`](../../wiki/entities/gencad-3d.md)
- 互链：[`wiki/entities/gencad.md`](../../wiki/entities/gencad.md)、[`wiki/concepts/text-to-cad.md`](../../wiki/concepts/text-to-cad.md)、[`wiki/entities/urdf-studio.md`](../../wiki/entities/urdf-studio.md)（CAD→机器人描述下游）

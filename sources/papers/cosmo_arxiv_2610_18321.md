# CoSMO: Conformal Semantic Mapping for Open-World Safe Robot Navigation（arXiv:2610.18321）

> 来源归档（ingest）

- **标题：** CoSMO: Conformal Semantic Mapping for Open-World Safe Robot Navigation
- **arXiv：** <https://arxiv.org/abs/2610.18321>（摘要页称已被 CoRL 2026 接收）
- **代码（官方仓库，已核查）：** <https://github.com/Yangming911/CoSMO> — 归档见 [`sources/repos/cosmo.md`](../repos/cosmo.md)
- **项目页：** <https://yangming911.github.io/cosmo-page/> — 归档见 [`sources/sites/cosmo-project.md`](../sites/cosmo-project.md)
- **作者：** Shuqi Wang, Yiwei Zheng, Pingrui Zhang, Yue Gao, Xiang Yin
- **机构（arXiv HTML 版）：** 上海交通大学（Wang、Zheng、Gao、Yin）；上海创智学院（Wang、Gao、Yin）；复旦大学（Zhang）
- **入库日期：** 2026-10-17
- **一句话说明：** 把 VLM 嵌入校准成带覆盖保证的标签集合，按类别膨胀障碍物；安全词典只增不减，阈值用延迟专家反馈在线更新；MP3D 语义碰撞率较纯几何基线相对降 61.6%。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 页面链接 Code 指向 `github.com/Yangming911/CoSMO`，另有 YouTube 视频；页面仍写论文 "Coming soon"（引用为暂定），而 arXiv 已有 2610.18321。
- **仓库核查（2026-10-17）：** MIT；仅仿真代码；不含 MP3D ObjectNav episodes、场景资产与任何检查点（CLIP、YOLOv7、MobileSAM、可选 Grounding DINO 均需自备）；README 引用处为占位。
- **结论：** **部分开源**（Habitat 仿真评测代码；Go2 真机部分未见）。本次未本地运行。

## 摘录 1：方法

- 栅格地图每个单元存 VLM（CLIP 类）视觉嵌入，标签经文本编码器查询；非一致性分数为嵌入空间余弦类距离，与词表大小无关。
- 安全词典在线扩张且只增不减；ACI（Adaptive Conformal Inference）用延迟专家反馈更新阈值。
- 覆盖集合给每个障碍物单元最坏情况安全半径，规划器使用膨胀后的地图。
- 理论：校准界（Prop. 1）、延迟下的规划期覆盖（Prop. 2）、逐步安全定理，长期平均界为 α + τΔ + 2ϵ（τ 延迟，Δ 分布漂移，ϵ 依赖偏差）。

**对 wiki 的映射：** 核心原理；[具身语义认知地图](../../wiki/concepts/embodied-semantic-cognitive-map.md)。

## 摘录 2：结果

- 仿真：2,195 条 Matterport3D ObjectNav episode，语义安全违例终止 episode。OneMap：SR 11.3%、SCR 25.5%；GCLIP Argmax：SR 18.4%、SCR 13.1%；CoSMO 开放词表：SR 13.8%、SCR 9.8%；CoSMO + Grounding DINO 专家：SR 16.5%、SCR 10.5%。61.6% 为 SCR 相对 OneMap 的降幅（25.5→9.8）。
- 项目页另列 Uniform Dilation（SR 8.7%、SCR 8.9%、SPL 0.053）与 CoSMO known vocabulary（SR 10.4%、SCR 9.4%）；延迟消融 330 episode、0–100 步延迟，SCR 在 10.0%–10.9%。
- 真机：Unitree Go2，两个室内场景共 30 次；感知用 GCLIP，延迟专家为 Grounding-DINO（约 350 ms 延迟，反馈延迟 128–448 ms）。SR 46.67%、SCR 16.67%，经验真实标签覆盖 90.73%（目标 90%）；GCLIP Argmax top-1 准确率 58.06%（不同指标，不可直接比较）。

**对 wiki 的映射：** 实验与评测；覆盖率不等于无碰撞。

## 摘录 3：作者自述局限

- 保证假设可靠专家，专家标签噪声的影响未形式化。
- 每步查询专家可能昂贵，作者建议自触发查询。
- MP3D 布局紧凑，膨胀可能堵门，降低 SR。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-cosmo.md`；仅新增实体页。

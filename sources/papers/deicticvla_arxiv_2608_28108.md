# DeicticVLA: Unifying Instruction Modes Based on Language and Deictic Gestures in a Single VLA（arXiv:2608.28108）

> 来源归档（ingest）

- **标题：** DeicticVLA: Unifying Instruction Modes Based on Language and Deictic Gestures in a Single VLA
- **短名：** DeicticVLA
- **类型：** paper / vla / deictic-gesture / multimodal-instruction
- **arXiv：** <https://arxiv.org/abs/2608.28108>
- **PDF：** <https://arxiv.org/pdf/2608.28108>
- **HTML：** <https://arxiv.org/html/2608.28108>
- **DOI：** <https://doi.org/10.48550/arXiv.2608.28108>
- **作者：** Kango Yanagida、Tatsuya Aoki、Yuichiro Yoshikawa、Takato Horii
- **机构：** 大阪大学系统创新系（Dept. of Systems Innovation, Graduate School of Engineering Science, The University of Osaka）；东京大学国际神经智能研究中心（International Research Center for Neurointelligence, The University of Tokyo）
- **通讯：** {yanagida.kango.z7m@ecs., aoki.tatsuya.es@, y.yoshikawa.es@, takato@sys.es.} osaka-u.ac.jp
- **资助：** JST Moonshot R&D JPMJMS2011；JST BOOST JPMJBS2402
- **提交日期：** 2026-08-28（v1）
- **入库日期：** 2026-09-09
- **一句话说明：** 将 LI / VLI / VI 三种指令模式规范化为「文本 prompt + 指示 mask」，在单一 π₀ VLA 上系统比较 RGB 视觉提示 vs 分通道 mask 提示与两阶段训练；真机三任务单策略支持全模式，未见类别 VLI/VI 100% vs LI 16.7%。

## 开源状态（步骤 2.5）

- **项目页核查（2026-09-09）：** 打开 <https://arxiv.org/abs/2608.28108> 与 HTML 全文页。arXiv「Code, Data and Media Associated with this Article」区 **无** GitHub / Hugging Face / 项目页链接；作者页与机构页未列出配套仓库。
- **结论：** **截至入库日未开源 / 无官方项目页。** 论文未写 "code will be released" 承诺句；复现需等作者发布或自行按 π₀ + SAM 2 管线实现。

## 摘录 1：问题与统一接口（§I、§III-B）

- **痛点：** 同类/相似物体并存时，纯语言 LI 要求用户写冗长指代表达，且 VLA 对空间关系与序数词 grounding 不可靠，易按视觉偏置选错目标。
- **三种用户模式：** LI（仅语言）、VLI（语言 + 指示手势）、VI（仅指示手势，固定 prompt `follow visual instruction`）。
- **指令规范化：** text-prompt completion + deictic gesture grounding（用户点击 → SAM 2 生成 mask）；canonical 表示为 `(ℓ, 𝕄)`，LI 时 mask 为零 mask。
- **Mask 角色：** Mask-T（抓取目标）、Mask-G（放置目标）、Mask-R（空间指代参照，如「this」）。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-deicticvla.md`](../../wiki/entities/paper-deicticvla.md)；回链 [VLA](../../wiki/methods/vla.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)、[π₀](../../wiki/entities/paper-pi0.md)。

## 摘录 2：四种 prompting 与两阶段训练（§III-C、§III-D）

- **骨干：** π₀ [2] 全参数微调；动作 chunk 输出空间不变。
- **RGB 视觉提示：** VP-Fade（IA-VLA 式背景 0.8 灰化）、VP-BBox（红框，线宽 1% 图像尺寸）。
- **分通道 mask 提示：** MP-Early（ViT 输入前融合）、MP-Late（ViT 输出后融合）；多 mask 像素级 max 合成单通道；Alpha-CLIP 式 elementwise 加和。
- **两阶段训练（2S）：** Stage1 仅 𝒟_LI（30k steps）→ Stage2 联合 𝒟_all（LI+VLI+VI，30k steps）；对照 1S（60k 联合）与 2S-NoLI（Stage2 不含 LI）。
- **仿真：** LIBERO-Object / Spatial / Goal 子集；mask 用仿真 GT 分割（排除 SAM 误差）；2×A100；60k steps 总量对齐。

**对 wiki 的映射：** 强调「canonical 上游统一 + prompting 下游可换 + 2S 决定 unseen layout 能否用 mask」。

## 摘录 3：仿真与真机结果（§IV、§V）

- **仿真 2S 分布内：** 四法 mean SR 94.1–95.6%；VP-BBox 最高 95.6%。
- **仿真 zero-shot：** Object-ZS 上 VP-BBox / MP-Late ΔSR 最高（VLI +29.2、VI +27.8 量级）；Spatial-ZS 上 **VP-BBox 明显领先**（VLI +16.0、VI +11.1，其余单位数）。
- **训练消融：** 1S 使 VP-Fade/BBox 的 Object-ZS ΔSR 近零；2S-NoLI 使 LI SR 降至 63.6–72.1%（遗忘），且损害 VP-BBox/MP-Late 的 zero-shot。
- **真机：** UR5e + Robotiq 2F-85（Fin-ray）+ 顶视/腕部相机；GELLO 遥操作 360 ep；选 **MP-Late**（避免 VP 红框/背景衰减干扰）；2S 15k+15k。
- **真机亮点（Table V）：** OrganizeToy NO-Category：VLI/VI **100% (12/12)** vs LI **16.7% (2/12)**；PutBlock TL3：VI **75.0%** vs LI **20.0%**；VC-Surface：VLI **87.5%** vs LI **60.0%**。

**对 wiki 的映射：** 用「三模式单策略 + 2S 保留 LI 防遗忘 + VP-BBox 仿真强 / MP-Late 真机稳」写选型读法。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-deicticvla.md`**：机构 osaka + u-tokyo、流程 Mermaid、仿真/真机表、源码运行时序图「不适用」。
- 交叉：[Point-VLA](https://arxiv.org/abs/2608.23138)（LI+VLI 无 VI）、GesVLA（手势 latent token，不同融合路径）、[OpenVLA](../../wiki/entities/paper-openvla.md) 等 VLA 基线语境。

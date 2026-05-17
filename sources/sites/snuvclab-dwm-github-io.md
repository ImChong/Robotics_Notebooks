# Dexterous World Models（snuvclab.github.io/dwm）

> 来源归档（ingest）

- **标题：** Dexterous World Models（DWM）
- **类型：** project site
- **官方入口：** <https://snuvclab.github.io/dwm/>
- **论文：** <https://arxiv.org/abs/2512.17907>
- **代码：** <https://github.com/snuvclab/dwm>
- **入库日期：** 2026-05-17
- **一句话说明：** CVPR 2026 项目主页：TL;DR、摘要、**合成 / 真实场景结果**、两条关键洞察（混合交互–静态配对数据；全掩码修复初始化学残差动力学）及 BibTeX；与 arXiv 叙述一致，适合快速核对**对外表述**与**演示视频**入口。

## 页面结构（检索自 2026-05-17 公开站点）

| 区块 | 内容要点 |
|------|----------|
| TL;DR | 场景–动作条件视频扩散：在**给定静态 3D 场景**里模拟**灵巧人体操作**引起的动态变化 |
| Method Overview | 将具身动作分解为**第一人称相机运动** \(\mathcal{C}_{1:F}\) 与**手操作轨迹** \(\mathcal{H}_{1:F}\)；沿 \(\mathcal{C}\) 分别渲染静态场景视频与仅手网格视频作为条件 |
| Key Insight #1 | **TRUMANS** 合成精确对齐三元组 + **Taste-Rob** 固定机位真实视频 + **HaMeR** 手网格，混合学 **loco–manipulation** 与真实动力学 |
| Key Insight #2 | **全 mask 修复扩散**近似恒等映射 → 用静态场景视频作导航基线 → 手条件专注**残差动力学** |

## 对 wiki 的映射

- [DWM（Dexterous World Models）](../../wiki/methods/dwm.md) — 对外一句话定义、流程图与数据混合策略的读者向归纳
- 技术细节与公式以 [sources/papers/dwm_arxiv_2512_17907.md](../papers/dwm_arxiv_2512_17907.md) 为准

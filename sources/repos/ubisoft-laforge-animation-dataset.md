# Ubisoft La Forge Animation Dataset（LAFAN1）

- **标题**: Ubisoft La Forge Animation Dataset ("LAFAN1")
- **类型**: dataset / repo（含评估脚本）
- **仓库**: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
- **关联论文**: Harvey et al., *Robust Motion In-betweening*, SIGGRAPH / ACM TOG 2020（README 内 BibTeX）
- **许可**: Creative Commons **Attribution-NonCommercial-NoDerivatives 4.0**（`license.txt`，README 明确）
- **收录日期**: 2026-05-15

## 一句话摘要

Ubisoft La Forge 发布的 **BVH** 动捕数据集（README 命名 **LAFAN1**），面向动作过渡 / 动画研究基准；体量约 **5 名被试、77 序列、约 49.6 万帧 @30Hz（README 称约 4.6 小时）**，大文件走 **Git LFS**。

## 为何值得保留

- **足式与全身动作多样性**：README 按主题列出障碍地形行走、舞蹈、跌倒起身、瞄准移动等，适合作为 **locomotion / recovery** 类算法的参考动作库。
- **工程生态位**：社区仓库（如 `wbc_fsm`）已将其作为 **MoCap → 重定向 → 策略训练** 链路的公开范例数据来源。
- **许可边界清晰**：**NC-ND** 对商业产品与衍生分发限制严格；集成到产品或二次发布数据前必须单独合规审查。

## 数据与代码要点（编译自 README）

- 动画数据主要在 `lafan1/lafan1.zip`；序列均为 **BVH**；命名约定 `[theme][take]_subjectID.bvh`。
- 克隆需 **git-lfs**，否则 zip 损坏会导致 `BadZipfile`。
- 提供 `evaluate.py` / `evaluate_test.py` 与若干基线（Zero-Velocity、LERP+SLERP 等）及 **L2Q / L2P / NPSS** 指标说明。

## 对 Wiki 的映射

- **wiki/entities/lafan1-dataset.md**：数据集实体页。
- **wiki/entities/wbc-fsm.md**：在 related 中显式链接 LAFAN1 实体页，便于读者从部署案例跳回数据来源与许可。

# Ubisoft La Forge Animation Dataset（LAFAN1）

- **标题**: Ubisoft La Forge Animation Dataset ("LAFAN1")
- **类型**: dataset / repo（含评估脚本）
- **仓库**: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
- **关联论文**: Harvey et al., [*Robust Motion In-Betweening*](../papers/robust_motion_inbetweening_acm_tog_2020.md), SIGGRAPH / ACM TOG 2020；[DOI](https://doi.org/10.1145/3386569.3392480)
- **许可**: Creative Commons **Attribution-NonCommercial-NoDerivatives 4.0**（`license.txt`，README 明确）
- **收录日期**: 2026-05-15（2026-11-07 复核 README 全文与评估代码）
- **版本**: master 分支（2026-11-07 通过 raw.githubusercontent.com 获取）
- **核对范围：** 通读 `README.md` 全文（数据统计、主题表、克隆 / LFS 说明、评估基线与 L2Q / L2P / NPSS 指标定义）、`license.txt` 首段（CC BY-NC-ND 4.0）、`evaluate.py` 全文、`lafan1/benchmarks.py` 的 `fast_npss` 与 `benchmark_interpolation` 前半（零速度与插值基线的构造）；`lafan1/extract.py` 仅核对函数与参数命名，`lafan1/utils.py`、`evaluate_test.py` 与 `lafan1.zip` 内 BVH 数据本身未读、未运行代码。
- **通读：** 已完成

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

## 2026-11-07 README 核对补充

- 拍摄时间：README 写明 **Shot in May 2017**；5 名被试、77 条序列、496,672 帧 @30 fps（约 4.6 小时）；`lafan1.zip` 约 137 MB，走 Git LFS，缺 LFS 会报 `BadZipfile`。
- 主题表（序列数）：Obstacles 17、Walk 12、Dance 8、Fall and get up 6、Aiming 5、Ground 5、Multiple actions 4、Run 4、Fight 3、Jumps 3、Push and stumble 3、Fight and sports 2、Push and fall 2、Sprint 2、Push 1（合计 77）。
- 评估：训练统计量在 50 帧窗口、步长 20 上计算；测试窗口取 Subject 5，每 40 帧采一个，共 **2232** 个；基线为 Zero-Velocity（重复最后一个种子帧）与 Interpolation（根位置 LERP、四元数 SLERP）；指标 L2Q、L2P、NPSS。代码依赖仅 numpy（开发环境 Python 3.7 + numpy 1.17.4）。
- `evaluate.py`：解压 zip 到 `output/BVH` → 用 subject1–4 计算并缓存 `train_stats.pkl` → 以窗口 65、步长 40 构建 Subject 5 测试集 → `benchmark_interpolation`（过渡长度 5/15/30/45，各带 10 帧过去上下文与 10 帧未来上下文，仅首帧作目标）→ 保存 `results.pkl`。

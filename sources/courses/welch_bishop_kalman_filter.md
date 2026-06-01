# Welch & Bishop — An Introduction to the Kalman Filter

> 来源归档（ingest）

- **标题：** An Introduction to the Kalman Filter（UNC 技术报告 / 教程）
- **类型：** course / tutorial
- **作者：** Greg Welch, Gary Bishop
- **入库日期：** 2026-06-01
- **链接：** [PDF 与 HTML 入口](https://www.cs.unc.edu/~welch/kalman/) · [kalmanfilter.net](https://www.kalmanfilter.net/)
- **一句话说明：** 最广泛使用的 KF 入门教程（非期刊，但为官方技术报告与作者维护站点），适合实现前对齐符号与直觉。

## 为什么值得保留

- 机器人从业者常 **先读 Welch & Bishop 再读 Kalman (1960)**；作为 `sources/papers/kalman_filter_ekf_primary_refs.md` 的配套教程层，避免与 wiki 正文重复推导。

## 核心摘录

- **结构：** 动机 → 离散 KF 完整递推 → 扩展讨论（非线性提示指向 EKF）→ 实现注意事项。
- **记号：** 明确 **先验** $\hat{x}_{k|k-1}$ / **后验** $\hat{x}_{k|k}$ 与协方差 $P_{k|k-1}, P_{k|k}$，与 [kalman-filter](../../wiki/formalizations/kalman-filter.md) 一致。
- **局限：** 非线性部分仅作展望；深入 EKF 需转 Gelb / Simon 或 MIT Underactuated。

## 对 wiki 的映射

- [kalman-filter](../../wiki/formalizations/kalman-filter.md)
- [ekf](../../wiki/formalizations/ekf.md)
- [state-estimation](../../wiki/concepts/state-estimation.md)

## 当前提炼状态

- [x] 教程定位与 wiki 映射
- [ ] 后续可补：与 Simon (2006) 符号对照表

# HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-offs

- **类型：** paper
- **作者：** Zhuoshi Pan、ZhiYuan Feng、Yukuo Cen、Honglin Lin、Junru Lu、Zhongxia Chen、Shuo Yang、H. Vicky Zhao、Jie Tang
- **原文：** <https://arxiv.org/html/2610.26770v2>（v2，2026-10-20）
- **项目页：** <https://pzs19.github.io/HarnessPro/>
- **代码：** <https://github.com/pzs19/HarnessPro>
- **入库日期：** 2026-11-02

## 摘要与映射

- 演化时不使用官方测试或标准答案：固定的 LLM 合成测试评估候选。对应 [HarnessPro](../../wiki/entities/paper-harnesspro.md) 的流程总览。
- 轨迹分析、规划、编辑、验证与 Pareto 筛选构成递归过程；lite、medium、heavy 分别代表不同的性能与成本偏好。
- GPT-5.5 + Codex 下，Terminal-Bench 2 的 heavy 对 RHO：80.8% 对 76.3%，每题成本 $0.72 对 $1.09。测试运行次数和计分口径见论文 §5.1–5.2。

## 阅读状态

已核对摘要、算法、实验表与项目资料；论文其余正文和附录尚待逐节通读，本次为草稿。

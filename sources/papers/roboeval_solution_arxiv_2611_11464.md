# Off-Policy RL with Online Inference Tuning: 1st Place Solution to the CoRL 2026 RoboEval Challenge

- **作者：** Ilia Larchenko（独立研究者）
- **论文：** https://arxiv.org/abs/2611.11464v2
- **原文：** https://arxiv.org/html/2611.11464v2
- **PDF：** https://arxiv.org/pdf/2611.11464v2
- **版本：** arXiv:2611.11464v2，2026-11-24 修订（v1 初稿 2026-11-08），技术报告（预印本）
- **项目：** https://ilialarchenko.com/projects/roboeval2026/
- **代码：** https://github.com/IliaLarchenko/RoboEvalSolution
- **权重：** https://huggingface.co/IliaLarchenko/RoboEvalSolution
- **入库日期：** 2026-11-24
- **核对范围：** arXiv v2 PDF 摘要、§1–9、表 1–6、图注、附录 A 的示范诊断与评测建议；与作者项目页、代码 README、主办方最终榜单交叉核对；未执行实验。
- **通读：** 已完成

## 核心摘录（归纳）

- §2：双 Franka、MuJoCo、8 个位置及姿态随机化任务。RDS 综合结果与行为，任务阶段缺失和失败都影响最终分数；行为指标在参考值处饱和。
- §3：采集与训练异步；价值预测随轨迹保存，按当前任务均值附近的 RDS 梯度分配逐步奖励，加入 completion shaping，以收集策略自己的价值预测为基线；GAE λ=0.95，AWR 采样与 RECAP 优势条件共同使用优势。每条 rollout 同时用于 Thompson 参数搜索。
- §4：图像、task id、本体状态和优势 token；V query 仅访问图像与任务，Q query 对动作块预测相对 V 的残差。推理共享 prefix，最多 6 个候选批量流匹配与 CFG，Q 选择，跨 chunk inpainting。
- §5–7：原示范存在漏阶段；用仿真特权状态生成随机化脚本轨迹并从失败分支恢复。作者报告约 20 万生成轨迹（有丢弃）、24 万余策略 rollout、约 600 GPU 小时；50k 步附近平台期，最终权重 0.65×step48250+0.35×step58000。
- §8：作者 500 新种子/任务的 RDS 0.990、成功率 99.6%；50 公布种子/任务 RDS 0.987、成功率 99.25%。11 月 12 日主办方另以每任务 1000 个未公开种子评出第一名 0.988、99.3%，见 https://robo-eval.github.io/challenge.html#results（晚于 v1；已收入 v2 §9 表 6）。
- §9：前三在 50 个公开种子下的 RDS 相差不超过 0.002；组织方让三支队伍的原样提交在每任务 1,000 个预先固定的相同未公开种子上复跑，再按 RDS 排名。第二 0.982、第三 0.978，按种子配对 bootstrap 的 95% 区间排除零。
- 附录 A：应清理漏阶段示范、重新审视指标饱和和均值/中位数差异、排除 reset 接触、增加扰动及恢复评测；这些是作者建议而非现行规则。

## 对 wiki 的映射

- [独立论文实体](../../wiki/entities/paper-roboeval-solution.md)
- [项目主页归档](../sites/roboeval-solution.md)
- [源码归档](../repos/roboeval-solution.md)

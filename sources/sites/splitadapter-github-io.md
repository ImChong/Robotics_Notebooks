# splitadapter.github.io（SplitAdapter 项目页）

> 来源归档（ingest）

- **标题：** SplitAdapter — Samsung Future Robot AI Group
- **类型：** site / project-page
- **官方入口：** <https://splitadapter.github.io/>
- **入库日期：** 2026-06-04
- **一句话说明：** SplitAdapter 论文配套站点：强调 **因子化负载感知适配**、**6 kg 零样本真机搬运**（含训练质量范围外）、**地面搬起（0 cm）** 等重载场景视频，以及相对基策略 / WM-FiLM 的鲁棒性对比叙事。

## 页面公开信息（检索自 2026-06-04）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://splitadapter.github.io/> |
| 论文 abs | <https://arxiv.org/abs/2606.03297> |
| arXiv HTML | <https://arxiv.org/html/2606.03297> |

## 站点核心主张（与论文一致）

- **任务：** 人形 **搬箱 loco-manipulation**——抬起、搬运、放置；载荷与搬放高度变化。
- **方法一句话：** 冻结预训练搬箱策略 + **物体/负载** 与 **动力学** 双上下文编码器 + **分裂世界模型** + **GRL 解耦** + **分层 FiLM**。
- **亮点演示（项目页文案）：**
  - 最高 **6 kg** 载荷运输，含 **挑战性地面搬起**；
  - **零样本** 真机部署；
  - 相对基策略与 **world-model FiLM** 基线在 **2/4/6 kg** 与 **0/30/60 cm** 高度下更高 **Full-task** 成功率，**重载增益最大**。
- **附加视频：** 不同质量运输、高度变化、大箱/亚克力箱、**人机递接** 等。

## 对 wiki 的映射

- [`wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md`](../../wiki/entities/paper-splitadapter-load-aware-loco-manipulation.md) — 方法栈与实验归纳页。
- [`sources/papers/splitadapter_arxiv_2606_03297.md`](../papers/splitadapter_arxiv_2606_03297.md) — arXiv 结构化摘录。

## BibTeX（站点 / arXiv 提供）

```bibtex
@misc{kang2026splitadapterloadawarehumanoidlocomanipulation,
      title={SplitAdapter: Load-Aware Humanoid Loco-Manipulation via Factorized Adaptation},
      author={Jeonguk Kang and Hanbyel Cho and Sanghyun Kang and Donghan Koo},
      year={2026},
      eprint={2606.03297},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.03297},
}
```

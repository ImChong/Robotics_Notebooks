# Neural Introspection Gating（arXiv:2608.10824）

> 来源归档（ingest）

- **标题：** Neural Introspection Gating for Adaptive KV-Cache Reuse in Vision-Language-Action Models
- **类型：** paper / vla / kv-cache / efficient-inference / libero / iros-2026
- **arXiv abs：** <https://arxiv.org/abs/2608.10824>
- **PDF：** <https://arxiv.org/pdf/2608.10824>
- **HTML：** <https://arxiv.org/html/2608.10824>
- **项目页：** <https://zjw4321.github.io/neural-introspection-gating-page/> — 归档见 [`sources/sites/neural-introspection-gating-github-io.md`](../sites/neural-introspection-gating-github-io.md)
- **代码：** 截至入库日项目页 GitHub 按钮禁用（`href="#"`）
- **机构：** 东京大学（The University of Tokyo）
- **作者：** Zhijie Wu、Kento Kawaharazuka、Kei Okada
- **发表 / 上传：** 2026-08（arXiv:2608.10824；Accepted at IROS 2026）
- **入库日期：** 2026-08-13
- **一句话说明：** 训练无关扩展 VLA-Cache：用动作解码 top-1/top-2 logit margin 作内省门控，低置信时失效 KV 缓存并全量重算，在 LIBERO-Long 上收回盲缓存掉点且保留约 80% 算力节省。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [neural-introspection-gating-page](https://zjw4321.github.io/neural-introspection-gating-page/) | 方法四阶段 + OpenVLA / OFT 表 |
| 相关基线 | VLA-Cache（论文引用 [23]） | 观测空间 patch 相似度缓存 |

## 开源状态（步骤 2.5，截至 2026-08-13）

- **确认未开源：** 项目页 GitHub 按钮 `aria-disabled` / `href="#"`；无 Hugging Face 权重链接。
- **处理：** wiki 写明「截至 2026-08-13 未开源」；源码运行时序图 **不适用**。

## 摘要级要点

- **问题：** VLA-Cache 仅依视觉静态启发式复用 KV，无法感知模型在抓取对齐等时刻的不确定性，陈旧缓存会污染自回归动作。
- **方法：** Gated VLA-Cache — 保留 static patch / task-relevant / entropy-adaptive 三阶段；新增均值 logit margin \(m_{t-1}\)；\(m<\theta_m\) 则全量 recompute。
- **结果要点（OpenVLA）：**
  - LIBERO-Long：**Full 54.0** / **VLA-Cache 50.2** / **Ours 54.8**（TFLOPs 1.88 / 1.43 / 1.54）
  - Goal：**77.2 / 74.0 / 77.4**；平均接近 Full（69.9 vs 70.1）且低于 Full 算力
  - OpenVLA-OFT：缓存已稳时门控几乎不伤成功率、开销很小
- **局限：** 依赖离散动作 token 的 softmax margin；阈值对阈值动作头需另定义置信信号；阈值阈值依赖 \(\theta_m\)。

## 对 wiki 的映射

- 新建实体页：[wiki/entities/paper-neural-introspection-gating.md](../../wiki/entities/paper-neural-introspection-gating.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)（若存在）

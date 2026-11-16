# Neural Introspection Gating（arXiv:2608.10824）

> 来源归档（ingest）

- **标题：** Neural Introspection Gating for Adaptive KV-Cache Reuse in Vision-Language-Action Models
- **类型：** paper / vla / kv-cache / efficient-inference / libero / iros-2026
- **arXiv abs：** <https://arxiv.org/abs/2608.10824>
- **PDF：** <https://arxiv.org/pdf/2608.10824>
- **HTML：** <https://arxiv.org/html/2608.10824>
- **项目页：** <https://zjw4321.github.io/neural-introspection-gating-page/> — 归档见 [`sources/sites/neural-introspection-gating-github-io.md`](../sites/neural-introspection-gating-github-io.md)
- **代码：** <https://github.com/zjw4321/gated-vla-cache> — 归档见 [`sources/repos/gated_vla_cache.md`](../repos/gated_vla_cache.md)
- **机构：** 东京大学（The University of Tokyo，JSK Robotics Laboratory）
- **作者：** Zhijie Wu、Kento Kawaharazuka、Kei Okada
- **发表 / 上传：** 2026-08-11（arXiv:2608.10824v1；Accepted at IROS 2026）
- **入库日期：** 2026-08-13
- **通读：** 已完成
- **核对范围：** `2608.10824.txt`（全文 §I–§VII 含 Eq. 1、Alg. 1、Tables I–II、Figs. 1–7，L1–660）及 `2608.10824.abs.html`（元数据与提交历史，L1–200）
- **一句话说明：** 训练无关扩展 VLA-Cache：用动作解码 top-1/top-2 logit margin 作内省门控，低置信时失效 KV 缓存并全量重算，在 LIBERO-Long 上收回盲缓存掉点且保留约 80% 算力节省（OpenVLA 54.0%→50.2%→54.8%，1.88→1.43→1.54 TFLOPs）。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [neural-introspection-gating-page](https://zjw4321.github.io/neural-introspection-gating-page/) | 方法四阶段 + OpenVLA / OFT 表 |
| 代码仓 | [zjw4321/gated-vla-cache](https://github.com/zjw4321/gated-vla-cache) | 官方 OpenVLA / OpenVLA-OFT 实现（Apache 2.0） |
| 相关基线 | VLA-Cache（论文引用 [4]） | 观测空间 patch 相似度缓存（\(\tau=0.996\)） |

## 开源状态（步骤 2.5，核实更新）

- **已开源（Apache-2.0）：** 项目页已开放指向 `https://github.com/zjw4321/gated-vla-cache` 的仓库，包含 OpenVLA 与 OpenVLA-OFT 两套可运行评测代码；权重需本地下载官方 OpenVLA / OpenVLA-OFT checkpoint。
- **处理：** wiki `## 源码运行时序图` 基于官方仓库实码绘制；`paper-neural-introspection-gating.md` 同步更新 `code` 与工程实践细节。

## 摘要级要点

- **问题：** VLA-Cache 仅依视觉静态启发式（阶段 1 像素余弦相似度 \(\tau=0.996\) + 阶段 2 text→vision 注意力过滤 `top-k=120` + 阶段 3 逐层注意力熵自适应复用）决定 KV 复用，无法感知模型在抓取对齐等时刻的不确定性，陈旧缓存会污染自回归动作。
- **方法：** Gated VLA-Cache — 保留 VLA-Cache 三阶段；利用前向副产品计算均值 top-1/top-2 概率 margin \(m_{t-1}=\frac{1}{D}\sum_{d=1}^{D}[p_1(a^d_{t-1})-p_2(a^d_{t-1})]\)；若 \(m_{t-1}<\theta_m\)（OpenVLA 标定为 0.65，OpenVLA-OFT 标定为 0.50），下步失效全部缓存并做全量前向重算（单步滞后，LIBERO-Long 约 24% 步数触发、76% 步数复用；Simple 任务 <10%）。
- **为何选 logit margin（§VI 对照消融）：**
  - 动作总熵被 32,000 词表长尾近零概率噪声淹没；
  - 注意力熵受深层自回归 LLM 的 **attention sink** 现象锚定，难反映视觉焦点切换；
  - 相邻步动作 delta 反映的是接近/抓取等**阶段切换**，而非模型内在不确定性。
- **结果要点（LIBERO 4 套件 × 50 episodes/任务 = 500 episodes/套件，NVIDIA A100）：**
  - **OpenVLA（Table I，SR% / TFLOPs）：** Spatial `78.8/1.89` → `78.8/1.43` → **`79.4/1.55`**；Object `70.4/1.86` → `69.4/1.44` → `67.8/1.54`；Goal `77.2/1.83` → `74.0/1.40` → **`77.4/1.50`**（收回 106%）；Long `54.0/1.88` → `50.2/1.43` → **`54.8/1.54`**（收回 121%）；Avg `70.1/1.87` → `68.1/1.43` → **`69.9/1.53`**（保留约 80% 算力节省，低于 Full 18%）。
  - **OpenVLA-OFT（Table II，8 动作 chunk，SR% / TFLOPs）：** Spatial `93.8/4.00` → `93.8/3.11` → **`94.0/3.23`**；Object `98.4/3.96` → `98.6/3.05` → `98.6/3.08`；Goal `97.2/3.94` → `96.8/3.06` → `97.0/3.07`；Long `93.8/3.98` → `94.0/3.05` → `93.0/3.13`；Avg `95.8/3.97` → `95.8/3.07` → `95.7/3.13`（缓存原本已稳，门控几乎休眠、无明显掉点）。
- **局限（§VI）：** (1) 无法区分「缓存引发的不确定性」与「任务固有模糊」（多算仅增算力不伤成功率）；(2) 存在一步滞后（用 \(t-1\) 步 margin 决定第 \(t\) 步缓存）；(3) 需按模型族在少量验证集上标定单阈值 \(\theta_m\)（OpenVLA 0.55–0.75、OFT 0.40–0.60 平台稳健）；(4) 仅在 LIBERO 仿真评测，尚未做真机验证。

## 对 wiki 的映射

- 更新实体页：[wiki/entities/paper-neural-introspection-gating.md](../../wiki/entities/paper-neural-introspection-gating.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)

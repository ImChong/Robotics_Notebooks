# giraffeguan.github.io/limmt（LIMMT 项目页）

> 来源归档（ingest）

- **标题：** LIMMT: Less Is More for Motion Tracking — ICML 2026
- **类型：** site / project-page
- **官方入口：** <https://giraffeguan.github.io/limmt/>
- **入库日期：** 2026-06-09
- **一句话说明：** 论文配套站点：强调 **GQS 三阶段数据策展**、**3% AMASS 胜全量**、**Any2Track / TWIST2 plug-and-play**、消融与 **PHUMA 跨域**、**Unitree G1 真机** 多类动作演示视频。

## 页面公开信息（检索自 2026-06-09）

| 资源 | URL |
|------|-----|
| 项目首页 | <https://giraffeguan.github.io/limmt/> |
| arXiv | <https://arxiv.org/abs/2606.06953> |

## 与论文一致的公开主张（便于 wiki 溯源）

1. **核心口号**：Curated Data Beats 100% Full Dataset；**3%** 精选 AMASS 优于全库。
2. **质量三维**：Physics feasibility · Diversity · Complexity。
3. **GQS 管线**：Stage I 仿真 $S_{phy}$ 过滤 → Stage II HME（PAE 分解 $A,F,\phi,b$）→ Stage III Global Weighted FPS + 复杂度偏置。
4. **主表（AMASS）**：Any2Track / TWIST2 上 GQS @ 3%/10% 全面优于 Full Data；Random 3% 灾难性下降。
5. **消融 @ 3%**：物理过滤最关键；多样性为覆盖前提；复杂度加权精炼 MPJPE。
6. **训练动力学**：GQS 10% 自早期即更高 reward / 更低 error（<0.5B steps）。
7. **PHUMA**：10% 子集 in-domain 与 cross-domain（→AMASS）均优于全量。
8. **真机**：G1 上 10% GQS 策略无微调；舞蹈 / 竞技 / 表现力等多段视频。

## 对 wiki 的映射

- [`wiki/methods/limmt-gqs-motion-curation.md`](../../wiki/methods/limmt-gqs-motion-curation.md) — 方法栈、实验表与真机归纳

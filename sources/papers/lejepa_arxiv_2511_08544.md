# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics（arXiv:2511.08544）

> 来源归档（ingest）

- **标题：** LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics
- **类型：** paper / JEPA / self-supervised / SIGReg
- **arXiv：** <https://arxiv.org/abs/2511.08544>
- **代码：** <https://github.com/rbalestr-lab/lejepa>（CC BY-NC 4.0）
- **作者：** Randall Balestriero、Yann LeCun
- **机构：** 布朗大学；纽约大学；Meta-FAIR
- **入库日期：** 2026-09-05
- **一句话说明：** 证明各向同性高斯是 JEPA embedding 的最优下游风险分布，并用 SIGReg（随机投影 + 特征函数检验）把坍塌从启发式换成可证明约束。

## 开源状态（2026-09-05）

- **已开源：** [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa)（1337★）。`lejepa` 包 + ImageNet 最小例（`MINIMAL.md`）。
- **许可证：** **CC BY-NC 4.0**（不可商用）。
- 项目页：仓库即入口；无独立 `.github.io`。

## 核心论文摘录（MVP）

### 1) 各向同性高斯是探针最优

- **链接：** §3
- **摘录要点：** 线性 / 非线性探针下，各向同性高斯唯一最小化下游预测风险；各向异性同时放大偏差与方差。
- **对 wiki 的映射：** [paper-lejepa](../../wiki/entities/paper-lejepa.md)

### 2) SIGReg 线性复杂度

- **链接：** §4
- **摘录要点：** 随机 1D 投影 + Epps–Pulley 等特征函数检验；时间/内存对维数与 batch 线性。
- **对 wiki 的映射：** [paper-lewm](../../wiki/entities/paper-lewm.md) 把 SIGReg 接到动作条件 WM。

### 3) 启发式卸掉

- **链接：** §5–6
- **摘录要点：** 无 stop-gradient / teacher–student / EMA；单超参；ViT-H/14 IN1K 冻结线性探针 **79%**；Galaxy10 域内预训练超过 DINOv2/v3 迁移。
- **对 wiki 的映射：** [paper-levjepa](../../wiki/entities/paper-levjepa.md)

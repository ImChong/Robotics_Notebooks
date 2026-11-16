# Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets（arXiv:2603.11634）

> 来源归档（paper）

- **标题：** Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets
- **类型：** paper
- **作者：** Sreevardhan Sirigiri, Nathan Samuel de Lara, Christopher Agia, Florian Shkurti, Fabio Ramos（悉尼大学 / 多伦多大学 / 斯坦福大学 / Allen Institute for AI / NVIDIA）
- **arXiv：** <https://arxiv.org/abs/2603.11634>（v1，2026-03-12，cs.RO；首页标注 Preprint under review）
- **PDF：** <https://arxiv.org/pdf/2603.11634v1>
- **代码：** 正文称有 "official repository"，但 v1 全文未给出具体仓库地址（未核实到开源仓库）
- **入库日期：** 2026-11-12
- **一句话说明：** 用「签名核（signature kernel）Gram 矩阵的香农 / 冯·诺依曼熵」度量示教轨迹数据集的多样性，并据此提出无需模型的贪心示教子集选择算法 FAKTUAL。
- **核对范围：** 通读 arXiv v1 PDF 正文第 I–IX 节、附录 A（示教转路径）、B/C（签名核与随机傅里叶特征近似，公式与复杂度仅略读）、D/E（子模性与子集算法）、F（任务与超参）、G（附加结果表 IV–VI、图 12）、I（引理证明，略读）；图 3–7 逐图查看；附录 H（通用 ML 多样性相关工作）仅浏览标题。未运行任何代码，未核对引用的参考文献原文，未核对论文 "official repository"（正文未给地址）。
- **通读：** 已完成

## 开源状态

- 论文提到 "official repository" 含配置文件，但 v1 全文未给 URL；未能确认开源仓库，wiki 页因此不写 `code:`。

## 核心摘录

1. **问题：** 机器人示教数据集由变长轨迹、多模态（状态 / 动作 / 视频）构成，缺少被广泛接受的多样性度量；需要尊重轨迹结构与几何的相似度（第 I 节）。
2. **度量（第 V 节，定义 3-4）：** 取归一化签名核 Gram 矩阵 K（对角线为 1），其特征值 λ 为 K/n 的特征值，Shannon 熵 H = -Σ λ ln λ，与冯·诺依曼熵 -Tr((K/n) ln(K/n)) 相等（附录 I 引理 1）；对其取指数即 Vendi 分数（有效秩）。另有体积度量 Δ = det(K)。
3. **路径构造（图 1，附录 A）：** 每个时间步把 RGB 图像嵌入 ViT 特征（或用低维物体表示），与状态、动作拼接为多元路径；也可对各模态分别算签名核再凸组合（方法 2，文中称效果未优于拼接）；可追加单调时间坐标（因签名对重参数化不变）。
4. **FAKTUAL（第 VI 节）：** 在预算 m 下贪心近似「熵最大 m 子集」（NP 难的最大覆盖问题的近似）；可选混合：比例 p 取熵最大子集，其余 m(1-p) 取行列式最大子集（建议 p 在 0.5–0.9，第 VI 节）；消融（第 VIII 节）却称 p 取 0.2–0.5 即有提升且 p=0 实践中表现好，两处说法不同。
5. **实验设置（第 VII 节，附录 F）：** RoboMimic Can / Square（PH 与 MH）、Transport（MH）用 BC-RNN（Can）或带扩散动作头的 BC（Square、Transport）；MetaWorld 的 Stick Push / Door Open / Shelf Place 与四个 SO-ARM101 真机任务用 SmolVLA 微调；每个设置 50 次 rollout × 20 个 checkpoint，取 checkpoint 最大值，3 个随机种子。
6. **结果（图 3、5、6，表 IV–VI）：** Can(PH) 30 条时 FAKTUAL 均值 0.76 对随机 0.63，全集 180 条为 0.90（表 V）；Square(PH) 60 条时 FAKTUAL 0.47 低于随机 0.55，120 / 150 条时 0.75 / 0.79 对 0.73 / 0.75；Transport(MH) 各预算成功率均 ≤0.20，且 60 条时 0.08 对 0.09；MetaWorld 差距小（Shelf Place 成功率约 0.07–0.10，表 IV）。真机（图 6）：Organize 80 条 34.7% 对随机 24.7%、全集（100 条）28.7%；Pick-and-place 40 条 48.0% 对 39.3%、全集（50 条）42.0%；Mug drag 90 条 37.3% 对 33.3%、全集（100 条）37.3%；Drawer open 10 条即 100%，随机与全集同为 100%。
7. **相关性（图 4）：** 不同子集的签名熵与成功率的 Pearson r 为 0.74 至 1.00（随机选择与 FAKTUAL 两组）；各点即不同大小的子集，样本点数很少。
8. **对比基线（图 7，附录 F-C）：** 在 RoboMimic 低维观测设置下与 DemInf、CUPID、Demo-SCORE、Oracle、Success Similarity、随机比较；作者自述 FAKTUAL 总体优于随机，但仍低于多数依赖 rollout / 质量信号的基线，最接近 Demo-SCORE 的区间。表 VI：低维 vanilla BC 下 Can(PH) 全集 180 条 84±4、随机 90 条 68±7、FAKTUAL 90 条 86±3；Can(MH) 74±6 / 62±2 / 74±7。
9. **作者承认的局限（第 IX 节、第 VII 节）：** 不保证选出高质量轨迹；最多样的子集可能含有害的轨迹；不建立示教与策略行为的因果关系；若策略容量不足以学多模态，促多样性可能有害；对 Drawer open 的五种光照没有区分能力（ViT 嵌入难捕获全局光照）；目标条件任务（Organize）未系统偏向某个终点；全局对齐核（GAK）选样与随机相当（第 VIII 节 c）；低示教预算下较难任务的多样性选样可能低于随机。
10. **效率声明的证据：** 摘要称「比近期机器人数据策展方法计算上高效得多」，正文只有表 I 的定性对比（是否需要辅助模型 / 策略 / 在线 rollout）与附录 B/C 的复杂度公式，v1 未见实测耗时对比表（核对于全文，未发现）。

## 对 wiki 的映射

- [paper-faktual-signature-entropy-diversity](../../wiki/entities/paper-faktual-signature-entropy-diversity.md)

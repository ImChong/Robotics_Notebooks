# AI Infra Guide｜公众号导读（用户提供 PDF）

- **类型：** blog / 外部导读
- **文章标题：** AI Infra Guide
- **作者署名：** 不知名树 · 古月居（按上传 PDF 页眉原样记录，具体署名关系未核验）
- **文章链接：** <https://mp.weixin.qq.com/s/jeH9oBxVab-AJd5Qcotr5A>
- **提供资料：** 用户上传 `AI Infra Guide.pdf`（2026-11-15 PDF 页面时间；共 14 页，其中核心正文及参考资料集中在第 1–11 页）
- **核对范围：** 已阅读 PDF 的导读、六章、结语、参考资料以及末尾页面；未对原公众号网页逐字核验
- **通读：** 已完成
- **定位：** 二手导读，不等于原始技术文档；官方原始项目是 <https://github.com/caomaolufei/AIInfraGuide>

## 六章结构摘要（按附件组织）

1. **计算、显存、通信：** GEMM、HBM 容量/带宽、AllReduce/All-to-All；瓶颈需要以数据证明。
2. **GPU 与 CUDA：** Warp / Block / Grid、Shared Memory、合并访存、分支发散；建议先写向量加法再做 Reduce/GEMM。
3. **算子优化：** Reduce / GEMM / Softmax 和 FlashAttention，强调 Tiling 与 Online Softmax 的 IO 优化。
4. **分布式训练：** DDP、ZeRO-1/2/3、FSDP、TP、PP、EP、CP；用通信换显存或并行吞吐。
5. **推理 Serving：** Prefill/Decode、KV Cache/PagedAttention、Continuous Batching、量化、Speculative Decoding 和 PD 分离。
6. **实验驱动学习：** Baseline → Benchmark → Profiling → 优化假设 → 重测；记录 GPU、Shape、精度、延迟、吞吐、显存。

## 易误读边界

- 附件是对技术链路的**概念性综述**，没有独立的系统 Benchmark/可复现速度实验，不能以此声称任何技术在指定设备上的固定加速比。
- 附件第 9 页的「6.2 每个阶段应该做到什么程度」未展示具体正文条目，学习标准以官方路线页为准。
- 附件部分源码示例因排版出现断字与横向截断；代码必须回原始仓库/官方文档核对，不能直接复制执行。

## 对 wiki 的映射

- [AIInfraGuide 实体节点](../../wiki/entities/aiinfraguide.md)
- [AIInfraGuide 官方仓库来源](../repos/aiinfraguide.md)

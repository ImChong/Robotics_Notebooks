# AIInfraGuide：AI Infra 全栈开源学习资料

- **类型：** repo / 开源技术教程
- **原始项目：** <https://github.com/caomaolufei/AIInfraGuide>
- **在线文档：** <https://caomaolufei.github.io/AIInfraGuide/>
- **官方学习路线：** <https://github.com/caomaolufei/AIInfraGuide/blob/main/docs/guides/AI%20Infra%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md>
- **核对版本：** 2026-11-15 查询 GitHub 主分支；官方学习路线 frontmatter 更新日 2026-11-01
- **收录日期：** 2026-11-15
- **许可：** MIT（官方 README 声明）
- **开源状态：** 网站源码和教程 Markdown **已公开**；不可将其等同于完整训练/推理引擎，CUDA 示例能否单独运行须逐章核实
- **核对范围：** 已核查官方 README、学习路线（前置/算子/分布式/推理四层及各阶段检验标准）、GEMM 章节概览、`package.json`、`astro.config.mjs`、`src/content/config.ts`、`src/pages/guides/[...slug].astro`、`src/pages/cuda/[...slug].astro`、`src/pages/index.astro` 首页入口。**尚未逐篇通读全部进阶章节与代码示例，未执行 GPU Benchmark**。
- **通读：** 未完成

## 一手资料概览

官方 README 将项目定位为从 GPU 硬件、CUDA 算子优化、分布式训练到推理服务的系统化教程，附面试宝典；性能分析板块在 README 中标记为规划中。四个学习模块分别为：前置基础、CUDA 编程与算子优化、分布式训练、推理优化。

官方学习路线将优化视为**计算 / 显存 / 通信**三类资源的交换；建议各阶段都留有可验证产物：独立训练循环、Reduce/GEMM/FlashAttention 调优、显存/通信预算、推理 QPS/延迟分位与显存验证等。

- **CUDA：** Grid / Block / Warp、合并访存、共享内存分块、Reduce/GEMM/Softmax、FlashAttention、Triton 与 profiling。
- **分布式：** DDP、ZeRO/FSDP、TP/PP/EP/CP 与 NCCL；优化时量化卡间/机间通信。
- **推理：** Prefill/Decode、KV Cache、PagedAttention、连续批处理、量化、推测解码与 PD 分离。
- **GEMM 第 4 章：** 当前读取到的是该章的分节概览（Baseline → Tiling → 向量化 → Bank Conflict → Double Buffer → cuBLAS 对照），**不把未读取的分节正文称为完成精读**。

## 官方网站源码入口（经核对）

- `npm run dev` → `astro dev`；`npm run build` → `astro check && astro build && pagefind --site dist ...`（见 `package.json`）。
- `src/content/config.ts`：Astro `glob` loader 从 `docs/guides/**/*.md` 装载 `guides` collection，并校验章节 frontmatter。
- `src/pages/guides/[...slug].astro`：`getStaticPaths()` 从 `getCollection('guides')` 生成非草稿的通用教程路径。
- `src/pages/cuda/[...slug].astro`：只选择 `cuda-optimization` 类别，传给 `GuideContent` 渲染。
- 这是**文档网站构建路径**；不能把它包装成 vLLM、PyTorch 或 CUDA Kernel 的运行调用栈。

## 对 wiki 的映射

- [AIInfraGuide 项目实体](../../wiki/entities/aiinfraguide.md)：技术全栈、资源权衡、工程练习、实际网站源码运行路径。
- [模型架构纵深路线 Stage 5](../../roadmap/depth-model-architecture.md)：规模/时延/显存预算与基准测试。
- [LLM 推理显存估算](../../wiki/concepts/llm-inference-vram-estimation.md)：KV Cache、显存占用与 TP 的专项深入。

## 待核验

- 分章 CUDA/PyTorch 示例的完整可运行入口、输入形状与性能数据；未在真实 GPU 上运行。
- 进阶章节、README 更新后的具体内容与文中经验阈值在不同 GPU 上的适用边界。

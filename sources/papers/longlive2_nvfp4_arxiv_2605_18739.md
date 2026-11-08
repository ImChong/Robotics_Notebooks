# LongLive-2.0：NVFP4 长视频生成训练与推理基础设施（论文原文归档）

- **英文题名：** LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation
- **类型：** paper
- **论文：** <https://arxiv.org/abs/2605.18739>
- **全文 HTML：** <https://arxiv.org/html/2605.18739v2>
- **论文版本：** arXiv:2605.18739v2，2026-05-19（v1：2026-05-18）
- **作者/机构：** Yukang Chen、Luozhou Wang、Wei Huang 等；NVIDIA
- **项目主页：** <https://nvlabs.github.io/LongLive/LongLive2/>
- **代码：** <https://github.com/NVlabs/LongLive/tree/main/LongLive2.0>
- **文档：** <https://nvlabs.github.io/LongLive/LongLive2/docs/>
- **核对范围：** 通读 arXiv v2 正文 §1–6、Table 1–5、附录 A–I（数据筛选、Balanced SP、SP 推理、attention sink、FP4 标度、量化消融、DMD 策略和实现参数）；比对官方项目页、主仓 README 与 LongLive2.0 使用文档、核心源码。
- **通读：** 已完成
- **核验日期：** 2026-11-08
- **沉淀到 wiki：** [LongLive-2.0 主实体](../../wiki/entities/paper-longlive-2-nvfp4-long-video-generation.md)

## 原文机制摘录

1. **Balanced SP (§2.1、Appendix C)：** 每个 SP rank 按时间 chunk 同时持有 clean/noisy latents；VAE 用左侧 halo 做局部精确编码；Ulysses All-to-All 后直接在交错顺序上构建 teacher-forcing block-sparse mask，不对 Q/K/V 做全局顺序恢复。保留 RoPE 全局索引并按全局有效 token 归一化损失。
2. **NVFP4 训练 (§2.2)：** E2M1 FP4 元素、每 16 值一组 FP8 E4M3 scale，加 tensor-level FP32 scale；线性层权重、激活与梯度采用相应低精度 GEMM，归约/归一化/优化器状态等保留较高精度。
3. **简化算法管线 (§4、Appendix H)：** 从 Wan2.2-TI2V-5B 直接训练多镜头 chunk-level AR；另在原始扩散模型上独立做 DMD LoRA 蒸馏（可与 AR 并行），将 few-step LoRA 注入 AR 模型；并非先 ODE 初始化再不断进行长视频调优。
4. **推理 (§3、§4.2)：** W4A4 模型、NVFP4 chunk KV cache、K-smoothing、并行反量化、滑动窗口 + global / per-shot sinks；scene cut 时重绑 shot sink。异步 VAE 把上一 chunk 解码和下一 chunk 去噪重叠。
5. **SP 推理（Appendix D）：** 理论设计支持使用低精度 KV / Q 降低 All-to-All 通信，论文报告约 3.6× 低位通信量；具体 release 中 `inference_sp.py` 当前会禁用 `kv_quant`，不能默认把论文机制当作发布脚本已全面支持。

## 论文结果（作者报告；非独立复测）

| 实验条件 | 结果 | 口径 |
|---|---|---|
| GB200 / 5B / 720p，BF16，4-step | 24.8 FPS；VBench Total 85.06 | FPS 排除 VAE |
| GB200 / 5B / 720p，NVFP4，4-step | 29.7 FPS；VBench Total 84.51 | FPS 排除 VAE |
| GB200 / 5B / 720p，NVFP4，2-step | **45.7 FPS；VBench Total 83.14** | FPS 排除 VAE；降步数有质量折衷 |
| GB200 / 5B / NVFP4+KV+异步解码，2-step，64s 视频 | **36.3 s E2E；19.4 GB 总峰值内存** | Table 3；异步 VAE 另用一张 GPU，不能当作纯单卡端到端 |
| 64s AR 训练，BF16+SP → NVFP4+Balanced SP | **1372.9 s → 639.5 s/iter** | Table 1；约 2.15×（表内四舍五入为 2.1×） |
| DMD 峰值显存 BF16 → NVFP4+LoRA | **70.5 → 49.0 GB/GPU** | Table 2；0.69× |
| VBench-Long 60s，LongLive-2.0 BF16 | **平均排名 3.67** | Table 5；与 4-step 长视频评测相对应 |
| 训练数据（Appendix B） | **120K** long videos | 16–32s、32–64s、>64s 各约 1/3 |

**重要限定：** 45.7 FPS 是扩散去噪阶段输出视频帧速率，不是完整 VAE 编码/解码、I/O 后的端到端 FPS。论文 GB200 E2E 64s=36.3s，另用 GPU 异步解码；H100 SP 表的 64s 数据注明由较短实测估计。速度、分辨率和模型规模不能与 1.3B / 480p 直接做等价质量比较。

## 论文局限与部署解读

- 45.7 FPS 档 VBench Total **83.14**，相比 4-step BF16 的 **85.06** 存在分数下降；不能只引用最快档。
- 论文宣称“首个”是作者限定的 **NVFP4 全训练+推理长视频系统**，并非“全球首个所有 4-bit 视频生成系统”的独立事实判定。
- 论文的 2-step、NVFP4、multi-shot、流式异步 VAE 依赖 checkpoint、并行设置及特定硬件条件；需按当前官方文档核对可复现路径。

## 对 wiki 的映射

- [唯一论文/项目/代码实体](../../wiki/entities/paper-longlive-2-nvfp4-long-video-generation.md)
- [源码实现核验](../repos/nvlabs-longlive2.md)
- [官方项目页](../sites/longlive-2-nvlabs.md)

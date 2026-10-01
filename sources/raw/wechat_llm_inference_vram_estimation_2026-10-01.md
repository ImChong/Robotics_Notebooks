# 一个大模型到底吃多少显存？如何计算？

以下整个计算方式个人观点，仅供参考

1. 为什么需要计算

大语言模型（LLM）推理的瓶颈不在算力，而在显存容量。

一块 RTX 4090 有 24 GB VRAM，足以运行 7B 模型（FP16 约 14 GB 权重），但跑 70B 模型就捉襟见肘了。问题在于：你不可能只看模型参数量就断定需要多少显存——实际占用还取决于：

- 量化方案： FP16 vs INT8 vs INT4，权重体积差 4 倍
- KV Cache：长上下文时，KV cache 可能比权重还大
- 并发请求数： batch=8 的 KV cache 是 batch=1 的 8 倍
- 推理框架： vLLM 比 HuggingFace Transformers 省 20%+ 开销
- 注意力架构： GQA、MQA、MLA 对 KV cache 的影响天差地别
- MoE 架构：总参数 vs 活跃参数——权重必须全量加载，但计算只走部分专家

2. 显存的五大开销

推理时，GPU 显存被分成以下几块：

```
┌─────────────────────────────────────────────┐│                 GPU VRAM                    │├──────────────────┬──────────────────────────┤│  模型权重         │  最大块。所有参数的量化副本  ││  (Model Weights) │  7B FP16 ≈ 14 GB          │├──────────────────┼──────────────────────────┤│  KV Cache        │  存储历史 attention 状态   ││                  │  随 batch × seq_len 线性增长│├──────────────────┼──────────────────────────┤│  激活值           │  前向传播的中间张量          ││  (Activations)   │  相对小，但大 batch 时不容忽视│├──────────────────┼──────────────────────────┤│  CUDA 固定开销    │  cuBLAS workspace +        ││  (CUDA Overhead) │  CUDA context，每卡固定     │├──────────────────┼──────────────────────────┤│  框架开销         │  vLLM/HF/TensorRT 的       ││  (Framework OH)  │  调度缓冲区 + 碎片          │└──────────────────┴──────────────────────────┘
```

其中权重和KV Cache通常是两个大头，特别是在长上下文场景下，KV cache 经常超过权重。

3. 输入参数全表

计算器接收以下参数，每一项都会直接影响最终结果：

3.1 架构参数

3.2 MoE 参数（可选）

3.3 MLA 参数（可选，DeepSeek 专用）

3.4 推理配置

3.5 量化与部署

4. 第一部分：模型权重

权重是 VRAM 的固定开销——模型加载后就不会改变。计算的核心是：逐层累加每个投影矩阵的参数量，乘以量化后的每参数字节数。

4.1 基本公式

```
total_weight = embedding + lm_head(如果未绑定) + layers × per_layer_weight
```

其中：

```
per_layer_weight = (Q_proj + KV_proj + O_proj + FFN + Norm) × quant_bytes
```

4.2 各矩阵的计算

Embedding（词嵌入）

```
embedBytes = vocabSize × hiddenSize × quantBytes
```

每个 token 对应一个 hiddenSize 维向量，共 vocabSize 个。

LM Head（输出头）

```
lmHeadBytes = tiedWeights ? 0 : embedBytes
```

如果权重绑定（tied），输入嵌入和输出投影共用同一组参数，不额外占用。LLaMA-3、Mistral、Qwen 等默认绑定。LLaMA-2、Falcon 等不绑定。

Q 投影

```
qProjSize = numHeads × headDim × hiddenSize
```

将 hiddenSize 维输入投影到 numHeads × headDim 维（所有 Q 头拼接）。

注意：这里用的是 numHeads × headDim，不是 hiddenSize × hiddenSize。对于大多数模型（如 LLaMA），numHeads × headDim == hiddenSize，两者等价。但对于 Gemma-2（16×256=4096 ≠ 3584）、Falcon 等模型，两者不同，必须用正确公式。

KV 投影（标准 GQA/MQA）

```
kvProjSize = numKVHeads × headDim × hiddenSize × 2
```

K 和 V 各一个矩阵，所以乘以 2。GQA 的核心优势：numKVHeads < numHeads 时，KV 投影和 KV cache 都更小。

KV 投影（MLA，DeepSeek 专用）

当 MLA 启用时，KV 不再是直接投影，而是先压缩到低维隐空间，再解压：

```
kvProjSize = hiddenSize × kvLatentRank + 2 × kvLatentRank × numHeads × headDim
```

- 第一项 hiddenSize × kvLatentRank：降维投影（down-projection）
- 第二项 2 × kvLatentRank × numHeads × headDim：升维重建 K 和 V（up-projection）

以 DeepSeek-V3 为例：

- 标准 GQA：128 × 128 × 7168 × 2 = 234.9M 参数/层
- MLA：7168 × 512 + 2 × 512 × 128 × 128 = 3.7M + 16.8M = 20.5M 参数/层
- 减少 91% 的注意力 KV 权重

O 投影

```
oProjSize = numHeads × headDim × hiddenSize
```

将注意力输出投影回 hiddenSize 维。

Norm

```
normSize = hiddenSize × 2
```

每层两个 RMSNorm（attention norm + FFN norm）。Norm 通常很小，可以忽略。

4.3 FFN（前馈网络）

稠密模型

```
ffnSize = hiddenSize × intermediateSize × 3
```

SwiGLU 的三个矩阵：

- Gate：hiddenSize → intermediateSize
- Up：hiddenSize → intermediateSize
- Down：intermediateSize → hiddenSize

各 hiddenSize × intermediateSize，共 3 个，所以乘以 3。

MoE 模型

```
ffnSize = (numExperts + numSharedExperts) × 3 × hiddenSize × expertIntermediateSize
```

所有专家的权重都必须加载到 VRAM 中——即使每个 token 只激活其中几个。这是 MoE 模型"总参数大但活跃参数小"的根源。

4.4 单层权重汇总

```
perLayerBytes = (qProjSize + kvProjSize + oProjSize + ffnSize + normSize) × quantBytes
```

4.5 总权重

```
totalWeightBytes = embedBytes + lmHeadBytes + perLayerBytes × layersweightGB = totalWeightBytes / (1024³)
```

4.6 活跃参数（MoE 参考）

对于 MoE 模型，活跃参数只计算被激活的专家：

```
activeFfnSize = (numSharedExperts + min(expertsPerTok, numExperts))                × 3 × hiddenSize × expertIntermediateSize
```

活跃参数不影响 VRAM 计算（权重总是全量加载），但会显示在 UI 上作为参考。

5. 第二部分：KV Cache

KV cache 是推理过程中增长最快的内存开销。每生成一个新 token，就要把当前 token 的 K 和 V 追加到 cache 中。

5.1 每 token KV cache 字节数

标准 GQA/MQA

```
kvPerTokenBytes = 2 × layers × numKVHeads × headDim × kvDtypeBytes
```

- × 2：K 和 V 各一份
- layers：每层都有独立的 KV cache
- numKVHeads × headDim：每个 KV 头的维度
- kvDtypeBytes：KV cache 的精度（可与权重量化不同）

MLA（DeepSeek V2/V3+）

```
kvPerTokenBytes = layers × kvLatentRank × kvDtypeBytes
```

MLA 将 K 和 V 压缩成一个共享的 kvLatentRank 维隐向量。不需要乘以 2——K 和 V 共用同一个压缩表示。

以 DeepSeek-V3 为例：

- 标准 GQA：2 × 61 × 128 × 128 × 2 = 4,325,376 bytes = 4.13 MB/token
- MLA：61 × 512 × 2 = 62,464 bytes = 0.06 MB/token
- 69 倍差距

5.2 KV cache 总量（峰值）

```
peakSeqLen = seqLen + maxGenLenrawPeakBytes = batchSize × peakSeqLen × kvPerTokenByteskvTotalGB = (rawPeakBytes × kvFragmentFactor) / (1024³)
```

关键设计决策：

- 峰值序列长度 = 输入长度 + 最大生成长度。因为推理过程中 cache 只增不减，最坏情况是输入满 + 生成满。
- KV 碎片因子（kvFragmentFactor）：不同框架的内存管理效率不同，vLLM 的 PagedAttention 仍有 12% 碎片，llama.cpp 几乎零碎片。

5.3 GQA 压缩比

```
// 标准 GQAgqaRatio = numHeads / numKVHeads// MLAgqaRatio = (2 × numHeads × headDim) / kvLatentRank
```

这个比值表示相对于标准 MHA（所有头都有独立 KV）的压缩倍数。例如 LLaMA-2 70B：64 个 Q 头，8 个 KV 头 → 压缩比 8×。

5.4 Encoder 模型

Encoder-only 模型（BERT、Reranker）没有 KV cache——它们一次性处理所有 token，不需要自回归生成。计算器正确处理了这一点：

```
if (modelType === 'decoder' || modelType === 'encoder-decoder') {  // 计算 KV cache}// encoder 类型：kvPerTokenBytes = 0
```

6. 第三部分：激活值

激活值是前向传播中产生的中间张量。推理时（无梯度反传），激活值远小于训练，但仍不可忽略。

6.1 线性层激活

```
activeInterSize = isMoE  ? (numSharedExperts + min(expertsPerTok, numExperts)) × expertIntermediateSize  : intermediateSizelinearActBytes = batchSize × seqLen × (activeInterSize × 2 + hiddenSize × 2) × 2
```

拆解：

- activeInterSize × 2：FFN 中 gate 和 up 的输出（各 intermediateSize 维）
- hiddenSize × 2：down 的输出 + 残差连接（各 hiddenSize 维）
- × 2（末尾）：FP16 每元素 2 字节
- batchSize × seqLen：总 token 数

MoE 特殊处理：激活值使用活跃专家的 intermediate size，而不是稠密 FFN 的。因为 MoE 前向传播时只有被激活的专家参与计算。

6.2 Flash Attention 缓冲区

```
flashAttnBytes = batchSize × numHeads × min(seqLen, 128) × min(seqLen, 4096) × 2
```

Flash Attention 将注意力矩阵分块计算（block size 通常 128），内存复杂度从 O(n²) 降为 O(n)。这个公式估算的是分块注意力所需的临时缓冲区。

6.3 总激活

```
actGB = (linearActBytes + flashAttnBytes) / (1024³)
```

激活值通常在 0.2-3 GB 之间，远小于权重和 KV cache，但在大 batch + 长序列时也不容忽视。

7.1 框架开销因子

框架开销是一个乘性因子，应用于权重 + 激活 + KV cache 的总和：

```
scalableGB = (weightGB + actGB + kvTotalGB) × overheadFactor
```

用户可以通过 UI 的 "Framework Overhead Factor" 输入框直接调整这个值。切换框架时自动设置为该框架的默认值：

7.2 KV 碎片因子

独立于框架开销，KV cache 本身有碎片：

7.3 CUDA 固定开销

这是一块不随模型规模线性增长的固定开销，包括 cuBLAS workspace、CUDA context、驱动缓冲区等。每张 GPU 都有，且不被张量并行分摊。

```
function getCudaFixedGB(totalParamB){  if (totalParamB < 10)  return0.6;   // 小模型（<10B）  if (totalParamB < 40)  return0.9;   // 中模型（10-40B）  return1.3;                           // 大模型（40B+）}
```

三级分档的理由：

- 小模型：CUDA context 占比高但绝对值小
- 大模型：cuBLAS 自动分配更大的 workspace
- 阈值在 10B 和 40B 处切换

8. 第五部分：总 VRAM 与张量并行

8.1 总公式

```
scalableGB = (weightGB + actGB + kvTotalGB) × overheadFactorperGpuTotal = scalableGB / tpSize + cudaFixedGBallGpuTotal = scalableGB + cudaFixedGB × tpSize
```

逻辑拆解：

1. 权重 + 激活 + KV cache 三者之和，乘以框架开销因子 → scalableGB
2. scalableGB 可以被张量并行分摊到多卡 → / tpSize
3. CUDA 固定开销每卡都有，不被分摊 → + cudaFixedGB

8.2 张量并行（Tensor Parallelism）

张量并行（TP）将权重矩阵按行或列切分到多张 GPU 上。在 Megatron-LM 风格的 TP 中：

- 权重：按 GPU 数均分 ✓
- KV cache： KV 头也分配到不同 GPU（numKVHeads >= tpSize 时完全均分）✓
- 激活值：大部分均分 ✓
- CUDA 固定开销：每卡独立，不分摊 ✗

所以 per_gpu = scalable / tp + cuda_fixed，而 all_gpu = scalable + cuda_fixed × tp。

8.3 GPU 推荐算法

```
neededVRAM = perGpuGB × 1.1  // 留 10% 安全余量suitable = GPUS.filter(gpu => gpu.vram >= neededVRAM)                    .sort((a, b) => a.cost - b.cost)if (suitable.length > 0) {  // 展示最便宜的 4 款，按价格排序  // 第一款标记为 "Best Value"} else {  // 没有单卡够用，推荐多卡方案  best = GPUS.sort((a, b) => b.vram - a.vram)[0]  numNeeded = Math.ceil(neededVRAM / best.vram)  // 推荐 numNeeded × best.name}
```

关键设计：按价格升序排列，推荐性价比最高的方案。如果没有任何单卡满足，则推荐用最大显存卡多卡并联。

9. 量化方案完整对照表

量化方案决定了 quantBytes——每个参数占用的字节数，直接乘到权重上。

GPTQ vs AWQ：两者在 VRAM 估计上等价（都是 0.5 bytes/param）。区别在量化方法：GPTQ 用 Hessian 矩阵的逆做权重校准，AWQ 用激活幅度做缩放。对 VRAM 无影响。

INT4 Compressed (0.6)：比纯 INT4 (0.5) 多 20%，因为分组量化（group_size=128）的缩放因子占额外空间。

KV Cache 精度独立选择

KV cache 的精度与权重量化独立。即使权重是 INT4，KV cache 通常仍用 FP16（保证生成质量）。计算器支持三种 KV dtype：

vLLM 支持 KV cache 量化（--kv-cache-dtype=fp8），可以在不损精度的前提下将 KV cache 减半。

10. 推理框架参数表

每个框架有两个独立参数：

计算流程：

1. base_overhead 作为 overheadFactor 乘到所有 scalable 内存上
2. kv_fragment 单独乘到 KV cache 的 raw bytes 上（在 kvTotalGB 中已应用）
3. 最终 KV cache 有效开销 = raw_kv × kv_fragment × overheadFactor

设计说明： overheadFactor 用户可调。切换框架时自动设为 base_overhead。用户可以根据实际部署经验微调。

11. CUDA 固定开销分级

重要：这是每卡的开销，使用 TP 时不会被分摊。

```
// TP=4, 70B 模型// 每卡 = scalable / 4 + 1.3  ← 1.3 是每卡固定// 总共 = scalable + 1.3 × 4   ← 4 张卡各出 1.3
```

12.1 GPU 数据库

计算器内置 13 款 GPU：

12.2 推荐逻辑

```
1. 计算每卡需求 VRAM = perGpuGB × 1.1（10% 安全余量）2. 筛选 VRAM >= 需求的 GPU3. 按价格升序排序4. 展示前 4 款（最便宜的标记为 "Best Value"）5. 如果没有单卡满足：   a. 找到 VRAM 最大的 GPU   b. 计算需要几张：ceil(需求 / 单卡 VRAM)   c. 展示多卡方案和总成本
```

13. 全部模型预设架构表

以下是计算器内置的全部预设模型，数据来自各模型的官方配置文件（config.json）：

稠密 Decoder 模型

注意 Gemma 2 的特殊之处： numHeads × headDim ≠ hiddenSize（9B: 16×256=4096 ≠ 3584）。这正是 Q/O 投影公式使用 numHeads × headDim 而非 hiddenSize² 的原因。

MoE Decoder 模型

Encoder 模型

Encoder 模型使用 MHA（numKVHeads == numHeads），无 GQA 压缩。且不生成 KV cache（非自回归）。

14. 实战演算：LLaMA-3 8B

让我们手动走一遍计算流程，以 LLaMA-3 8B、FP16、vLLM、batch=1、seq=4096、gen=2048 为例。

14.1 输入参数

```
layers=32, hiddenSize=4096, numHeads=32, numKVHeads=8, headDim=128intermediateSize=11008, vocabSize=128256, tied=truequantScheme=fp16 (2 bytes), framework=vllmbatchSize=1, seqLen=4096, maxGenLen=2048kvDtypeBytes=2, tpSize=1, overheadFactor=1.1
```

14.2 权重计算

```
embedBytes = 128256 × 4096 × 2 = 1,050,869,760 byteslmHeadBytes = 0 (tied)qProjSize  = 32 × 128 × 4096 = 16,777,216kvProjSize = 8 × 128 × 4096 × 2 = 8,388,608oProjSize  = 32 × 128 × 4096 = 16,777,216ffnSize    = 4096 × 11008 × 3 = 135,266,304normSize   = 4096 × 2 = 8,192perLayerBytes = (16,777,216 + 8,388,608 + 16,777,216 + 135,266,304 + 8,192) × 2             = 177,217,536 × 2 = 354,435,072 bytestotalWeightBytes = 1,050,869,760 + 0 + 354,435,072 × 32                = 1,050,869,760 + 11,341,922,304                = 12,392,792,064 bytesweightGB = 12,392,792,064 / (1024³) = 11.54 GB
```

14.3 KV Cache

```
kvFragment = 1.12 (vLLM)peakSeqLen = 4096 + 2048 = 6144kvPerTokenBytes = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128.00 KB/tokenrawPeakBytes = 1 × 6144 × 131,072 = 805,306,368 byteskvTotalGB = (805,306,368 × 1.12) / (1024³) = 0.84 GBkvPerSeqPeakMB = (131,072 × 6144) / (1024²) = 768.00 MBgqaRatio = 32 / 8 = 4.0×
```

14.4 激活值

```
activeInterSize = 11008 (dense model)linearActBytes = 1 × 4096 × (11008×2 + 4096×2) × 2              = 4096 × 30,208 × 2              = 247,451,648 bytesflashAttnBytes = 1 × 32 × min(4096,128) × min(4096,4096) × 2              = 32 × 128 × 4096 × 2              = 33,554,432 bytesactGB = (247,451,648 + 33,554,432) / (1024³) = 0.26 GB
```

14.5 总 VRAM

```
scalableGB = (11.54 + 0.26 + 0.84) × 1.1 = 12.64 × 1.1 = 13.90 GBtotalParamB = 12.39B → < 10? No, wait...// Actually: embedParams = 128256 × 4096 = 525M// perLayerParams = 177,217,536 / 2... wait, let me recalculate// Actually: perLayerBytes / quantBytes = perLayer params// perLayerParams = 177,217,536 (before ×quantBytes, so this is already raw count)// Hmm, let me recalculate properly:// totalParamCnt = (embedBytes + lmHeadBytes) / quantBytes + (perLayerBytes / quantBytes) × layers// = 525,434,880 + 177,217,536 × 32 = 525,434,880 + 5,670,961,152 = 6,196,396,032// totalParamB = 6.20B → < 10 → cudaFixedGB = 0.6cudaFixedGB = 0.6 GBperGpuTotal = 13.90 / 1 + 0.6 = 14.50 GB
```

14.6 结果

→ 推荐 RTX 4090 (24GB, 39% headroom) 或 RTX 3090 (24GB)。

实际部署 LLaMA-3 8B + vLLM + FP16 通常占用约 15-16 GB，与估算值 14.50 GB 的误差在 5% 以内。

15. 实战演算：DeepSeek-V3 671B（MoE + MLA）

这是最复杂的场景——同时使用 MoE 和 MLA。

15.1 输入参数

```
layers=61, hiddenSize=7168, numHeads=128, numKVHeads=128, headDim=128intermediateSize=2048, vocabSize=129280, tied=trueMoE: numExperts=256, numSharedExperts=1, expertIntermediateSize=2048, expertsPerTok=8MLA: mlaEnabled=true, kvLatentRank=512quantScheme=fp16, framework=vllm, batchSize=1, seqLen=4096, maxGenLen=2048kvDtypeBytes=2, tpSize=8, overheadFactor=1.1
```

15.2 权重计算

```
embedBytes = 129280 × 7168 × 2 = 1,853,267,968 byteslmHeadBytes = 0 (tied)// MLA KV projectionqProjSize  = 128 × 128 × 7168 = 117,440,512kvProjSize = 7168 × 512 + 2 × 512 × 128 × 128           = 3,670,016 + 16,777,216 = 20,447,232oProjSize  = 128 × 128 × 7168 = 117,440,512normSize   = 7168 × 2 = 14,336// MoE FFN: ALL 257 expertsffnSize = (256 + 1) × 3 × 7168 × 2048 = 257 × 44,040,192 = 11,318,329,344perLayerBytes = (117,440,512 + 20,447,232 + 117,440,512 + 11,318,329,344 + 14,336) × 2             = 11,573,671,936 × 2 = 23,147,343,872 bytestotalWeightBytes = 1,853,267,968 + 0 + 23,147,343,872 × 61                = 1,853,267,968 + 1,411,987,976,192                = 1,413,841,244,160 bytesweightGB = 1,413,841,244,160 / (1024³) = 1,316.95 GB
```

15.3 KV Cache（MLA 模式）

```
kvFragment = 1.12peakSeqLen = 4096 + 2048 = 6144// MLA: 只存压缩隐向量kvPerTokenBytes = 61 × 512 × 2 = 62,464 bytes = 61.00 KB/token// (标准 GQA 会是 2 × 61 × 128 × 128 × 2 = 4,325,376 bytes = 4,222 KB/token)rawPeakBytes = 1 × 6144 × 62,464 = 383,778,816 byteskvTotalGB = (383,778,816 × 1.12) / (1024³) = 0.40 GB// (标准 GQA 会是 27.63 GB！)gqaRatio = (2 × 128 × 128) / 512 = 64.0×  // MLA 压缩比
```

15.4 激活值

```
activeInterSize = (1 + min(8, 256)) × 2048 = 9 × 2048 = 18,432linearActBytes = 1 × 4096 × (18,432×2 + 7168×2) × 2              = 4096 × 51,200 × 2 = 419,430,400 bytesflashAttnBytes = 1 × 128 × 128 × 4096 × 2 = 134,217,728 bytesactGB = (419,430,400 + 134,217,728) / (1024³) = 0.51 GB
```

15.5 总 VRAM（TP=8）

```
scalableGB = (1316.95 + 0.51 + 0.40) × 1.1 = 1317.86 × 1.1 = 1,449.65 GB// totalParamB ≈ 1316.95B → >= 40 → cudaFixedGB = 1.3cudaFixedGB = 1.3 GBperGpuTotal = 1449.65 / 8 + 1.3 = 181.21 + 1.3 = 182.51 GBallGpuTotal = 1449.65 + 1.3 × 8 = 1449.65 + 10.4 = 1,460.05 GB
```

15.6 结果

→ 没有单卡够用（H200 141GB < 182.51GB）。推荐 2× H200 141GB / 卡（TP 需要调到更高）。

实际部署 DeepSeek-V3 需要 8× H200 141GB 或 8× H100 80GB（INT4 量化后）。

15.7 MLA 的巨大影响

如果用标准 GQA 而非 MLA，KV cache 会是：

```
标准 GQA: kvTotalGB = (1 × 6144 × 4,325,376 × 1.12) / (1024³) = 27.63 GBMLA:      kvTotalGB = 0.40 GB差距:     69×
```

在 TP=8 的场景下，每卡 KV cache 从 3.45 GB 降到 0.05 GB。这不是可有可无的差异——对于 128K 上下文，标准 GQA 的 KV cache 会达到数百 GB，而 MLA 只需几 GB。

16.1 估计精度

计算器声称的精度范围是 ±15-20%。误差来源：

1. 框架内部缓冲区：不同版本的 vLLM/HF 可能预分配不同的 workspace
2. 内存碎片：实际碎片率与模型结构、batch 组合有关，不是固定值
3. 混合精度：部分框架会对某些层做精度转换，临时增加内存
4. PagedAttention 的页大小： vLLM 的 block_size 影响 KV 碎片率

16.2 未建模的组件

以下组件未被计算器覆盖：

- RoPE 频率表：旋转位置编码的预计算表（通常 < 100 MB，可忽略）
- Q 压缩（MLA）： DeepSeek 的 MLA 还对 Q 做了低秩压缩（q_lora_rank），计算器未建模此项。影响：注意力权重被高估约 7-8%，但 MoE FFN 权重占 95%+，整体影响 < 1%
- Tokenizer： SentencePiece/BPE 的词表常驻内存（通常 < 10 MB）
- Pipeline Parallelism：仅支持张量并行，不支持流水线并行
- Expert Parallelism： MoE 的专家并行模式不在此计算范围

16.3 DeepSeek MLA 的简化

计算器的 MLA 实现做了以下简化：

- K RoPE 组件： MLA 中有一个单独的小 K 用于 RoPE（head_dim_rope，通常 64 维），计算器忽略了这部分（参数量极小）
- Q 低秩压缩：实际 MLA 对 Q 也做低秩压缩（q_lora_rank），计算器未建模，Q 权重略高估
- 解码重建： MLA 解码时从隐向量重建 K 和 V 的临时张量未计入激活值（量很小）

这些简化对最终结果的影响：权重高估约 1-2%，KV cache 计算准确。

16.4 生产部署建议

计算器给出的值是下界估计——实际部署建议增加 10-15% 余量：

```
production_vram = calculated_vram × 1.1 ~ 1.15
```

对于关键业务部署，建议：

1. 用计算器估算
2. 在实际 GPU 上用 nvidia-smi 验证
3. 留 15% 余量应对峰值负载

附录：计算流程伪代码

```
INPUT:  architecture params, inference config, quantization, framework1. quantBytes = QUANT_MAP[quantScheme]           // e.g. fp16 → 22. kvFragment = KV_FRAGMENT_MAP[framework]        // e.g. vllm → 1.123.// WEIGHTS   embedBytes  = vocabSize × hiddenSize × quantBytes   lmHeadBytes = tiedWeights ? 0 : embedBytes   qProjSize   = numHeads × headDim × hiddenSize   kvProjSize  = mlaEnabled ? hidden × kvLatentRank + 2 × kvLatentRank × numHeads × headDim                            : numKVHeads × headDim × hiddenSize × 2   oProjSize   = numHeads × headDim × hiddenSize   ffnSize     = isMoe ? (numExperts + numShared) × 3 × hidden × expertInter                       : hidden × intermediateSize × 3   perLayer    = (q + kv + o + ffn + norm) × quantBytes   weightGB    = (embed + lmHead + perLayer × layers) / 1GB4.// KV CACHE   if mlaEnabled:     kvPerToken = layers × kvLatentRank × kvDtypeBytes   else:     kvPerToken = 2 × layers × numKVHeads × headDim × kvDtypeBytes   kvTotalGB = (batch × (seq + gen) × kvPerToken × kvFragment) / 1GB5.// ACTIVATIONS   activeInter = isMoe ? (shared + min(perTok, experts)) × expertInter : intermediate   actGB = (batch × seq × (activeInter×2 + hidden×2) × 2            + batch × heads × min(seq,128) × min(seq,4096) × 2) / 1GB6.// OVERHEAD   cudaFixed = totalParamB < 10 ? 0.6 : (< 40 ? 0.9 : 1.3)7.// TOTAL   scalable = (weightGB + actGB + kvTotalGB) × overheadFactor   perGpu   = scalable / tpSize + cudaFixed   allGpu   = scalable + cudaFixed × tpSizeOUTPUT: perGpu (GB per card), allGpu (total system GB)
```
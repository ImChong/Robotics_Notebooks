# LongCat-2.5 昇腾 W4A4 EPD 部署源码

- **官方代码：** <https://github.com/meituan-longcat/SGLang-FluentLLM/tree/longcat-2.5-npu>
- **官方运行说明：** <https://github.com/meituan-longcat/SGLang-FluentLLM/blob/longcat-2.5-npu/npu_test/ReadMe.md>
- **入口脚本（Encoder）：** [run_encoder_epd.sh](https://github.com/meituan-longcat/SGLang-FluentLLM/blob/longcat-2.5-npu/npu_test/flash26b/run_encoder_epd.sh)
- **入口脚本（P/D 调度）：** [run_pro_w4a4kv8_mm_epd.sh](https://github.com/meituan-longcat/SGLang-FluentLLM/blob/longcat-2.5-npu/npu_test/flash26b/run_pro_w4a4kv8_mm_epd.sh)
- **节点角色脚本：** [pdmode_run.sh](https://github.com/meituan-longcat/SGLang-FluentLLM/blob/longcat-2.5-npu/npu_test/flash26b/pdmode_run.sh)
- **类型：** 官方开源 NPU 推理部署实现
- **入库日期：** 2026-11-10
- **核对范围：** npu_test/ReadMe.md 全文与上述三份 Bash 启动/路由脚本；核对硬件编排、配置校验、服务角色和调用入口，未实际启动硬件集群。
- **通读：** 已完成

## 真实入口与工作流

1. 先装 CANN、ATB、torch_npu、自定义 flash 算子与 Python 依赖，准备相互兼容的版本；模型和 Encoder 权重须分别就位。
2. Encoder 节点执行 `bash npu_test/flash26b/run_encoder_epd.sh`；脚本最终 `python -m sglang.srt.entrypoints.encoder_server`，指定 `--disaggregation-mode encode`、`--encoder-output-format rvq_indices`，默认 HTTP 9101、TP=2。
3. 调度端设置 `MODEL_PATH`、`ENCODER_URLS` 与 12 台机器的 `iplist`（**前 8 Decode，后 4 Prefill**），依次 `code`（rsync 同步）、`start`；建议先 `DRY_RUN=1` 核对。
4. `run_pro_w4a4kv8_mm_epd.sh` 通过 SSH 将配置交给 `pdmode_run.sh`；后者按机器序号启动 `run_decode.sh` 或 `run_prefill.sh`，最后一个 Prefill 节点还启动 `run_minilb.sh`。
5. P/D 端使用 `--enable-multimodal --language-only --encoder-urls ...`，视觉经独立 Encoder 计算，P/D 运行语言推理。MiniLB 默认端口 8081，开放 `/v1/chat/completions`。

## 厂商样例配置

| 分组 | 规模 |
|---|---|
| Ascend Atlas A2 总量 | **194 卡** |
| Encoder | **2 卡** |
| Prefill | **64 卡**（4 台 × 16 卡） |
| Decode | **128 卡**（8 台 × 16 卡） |

P/D 还包含 W4A4、KV INT8、NEXTN MTP、专家并行等引擎开关。此处为官方样例的**集群配置要求**，并非单机低显存可运行证明。

## 开源结论与局限

- **已开源：** 本分支的部署脚本和推理引擎；W4A4 模型权重已发布。
- **未验证：** 本地缺少 Ascend 194 卡集群与依赖环境，未执行服务启动、压力测试、性能复测。
- **不能推断：** 有此推理入口并不证明 LongCat-2.5 完整训练代码/数据公开。

**对应 wiki：** [LongCat-2.5](../../wiki/entities/longcat-2-5.md)

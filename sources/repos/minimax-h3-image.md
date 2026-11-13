# MiniMax-H3-Image：官方模型卡、推理仓库与许可证

> 一手来源归档（官方 GitHub / Hugging Face）

- **标题：** MiniMax-H3-Image — open-weight model for 16-megapixel image generation and editing
- **类型：** repo / model card / inference examples
- **机构：** MiniMax（稀宇科技）
- **代码/示例：** <https://github.com/MiniMax-AI/MiniMax-H3-Image>
- **模型卡：** <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image>
- **可选提示词扩展：** <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE>
- **基础编码器权重：** <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- **许可原文：** <https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/LICENSE>
- **入库/核查日期：** 2026-11-13
- **核对范围：** 通读官方 GitHub README（架构、权重、部署、请求参数、尺寸、完整样例、提示词与许可）、官方 HF 模型卡对应说明及 LICENSE 的地域/商用条款；逐项核对仓内 `examples/context_ir.py`、`examples/t2i.py`、`examples/edit.py` 的入口、请求路径、返回值及文件读写。仅进行源码静态核验；**没有下载约百 GB 权重、执行服务或复现生成速度**，未宣称通读第三方 SGLang/vLLM 全部源码。
- **通读：** 已完成
- **开源核查：** **开放权重 + 官方辅助脚本**（GitHub/HF 已可获取），具体部署依赖 SGLang、vLLM-Omni、Diffusers 等外部框架；许可证为 **MiniMax-H3-Image Community License**，不是 Apache/MIT/OSI 通用开源许可。

## 模型结构（原始模型卡）

| 模块 | 官方配置 |
|---|---|
| 条件编码 | 复用 H3 的 Qwen3-VL 文本编码器，第 50 层隐藏状态输入 DiT |
| 图像 VAE | CNN encoder / 24 层 2D ViT decoder；16× 空间下采样、24 latent 通道 |
| 主干 | 33.1B 参数单流 DiT、50 层，5376 hidden，56 heads；去掉 H3 音频分支 |
| 条件打包 | 文本与参考图编码后的 token 打包成序列，图像 VAE latent 进入统一生成 |
| 采样 | 6 NFE、无需 CFG（guidance_scale=1），使用 PDD 逐步输出头和 schedule |
| 输出 | 多宽高比约 16MP，4096×4096（1:1）、6272×2688（21:9）；无需超分器 |

- `transformer/` 66.28 GB、`vae/` 2.34 GB；编码器另复用 H3 仓库 `text_encoder/`（约 66.7 GB）。官方文档称原始推理检查点及编码器总共约 135 GB；HF 页面还包含 Diffusers 格式文件，完整下载量更大。
- 另外提供可选 **PE**（Prompt Expander，17.9 GB，NVFP4），**不是**主 DiT；MiniMax 托管 **H3-Image-Context-IR** 是另一套提示增强 API 流程。

## 官方示例入口与实际 I/O

| 文件 | 输入/步骤 | 输出 |
|---|---|---|
| [`examples/context_ir.py`](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/examples/context_ir.py) | 读取 `instruction.t2i.txt` / `instruction.edit.txt`，向 `/v1/h3_image_context_ir` 发送请求，编辑任务含参考图 Data URI | `revised_prompt.txt`，可选 `revised_size.txt` |
| [`examples/t2i.py`](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/examples/t2i.py) | 读取 `request.t2i.json`，可替换 `--prompt` / `--size`，POST 本地 SGLang `/v1/images/generations` | 解码 `data[0].b64_json` 写入 `t2i.png` |
| [`examples/edit.py`](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/examples/edit.py) | prompt 文件、`--image`（可重复）、宽高；multipart POST 本地 `/v1/images/edits` | 解码 `data[0].b64_json` 写入 `edit.png` |

官方 README 给出 `sglang serve --model-path MiniMaxAI/MiniMax-H3-Image --host 0.0.0.0 --port 30000 --warmup-mode off` 等服务启动方式；这属于**上游文档支持的入口**，不是本次运行成功记录。API 的 16 图上限、服务端默认提示增强与本地纯生成请求应分别理解。

## 许可关键限制（原文 2026-11-10）

- **适用地域排除**：欧盟、英国、韩国、美国；地域外使用受许可证约束，需向官方查询授权。
- **额外书面授权**：年化收入超过 **100 万美元**，或为第三方提供托管服务，需提前向 MiniMax 申请授权。
- **使用限制**：条款还包含输出/衍生模型和训练其他 AI 模型的限制；不要把“开权重”写成“无限制开源/免费商用”。
- 以上为条款摘要，非法律意见；具体以完整 [LICENSE](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/LICENSE) 为准。

## 对知识库的映射

- [MiniMax H3-Image 独立实体](../../wiki/entities/minimax-h3-image.md)
- [MiniMax H3 基础视频/音频模型](../../wiki/entities/minimax-h3.md)
- [Qwen-Image-2.1](../../wiki/entities/qwen-image-2-1.md)（同类图像生成模型选型比较）

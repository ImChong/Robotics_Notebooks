# FreeToken（边缘原生 MoE 推理）

> 来源归档（ingest）

- **标题：** FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.16157>
- **机构：** 加州大学伯克利分校（UC Berkeley）；麻省理工学院（MIT）；FlashML
- **作者：** Shuo Yang、Xiaoze Fan、Melissa Pan、Haocheng Xi、Zhe Wang、Shanlin Sun、Kurt Keutzer、Song Han、Matei Zaharia、Chenfeng Xu、Ion Stoica
- **项目页：** <https://flashml.ai>
- **代码：** <https://github.com/FlashML-org/FreeToken>
- **支持模型列表：** <https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md>
- **入库日期：** 2026-09-15
- **一句话说明：** 把个人电脑当作统一弹性推理平台，用带宽自适应 CPU–GPU 协同、专家缓存与语义锚点 KV 复用，在 8GB 笔电到单卡工作站上本地跑 35B–753B 级前沿 MoE；Apache-2.0 已开源。

## 核心摘录（MVP）

### 1) 问题：数据中心假设 vs 边缘现实

- **摘录要点：** 前沿开放权重 MoE 已普及，但 serving 栈仍默认数据中心 GPU 池。边缘有两条硬约束：**agent 负载的执行模式持续变化**（工具调用、思考块、上下文编辑）；**单机异构资源配比因机器而异**（VRAM、PCIe、CPU 核、主机内存）。固定 offload 策略无法同时适配笔电 8GB GPU 与台式 5090。
- **对 wiki 的映射：**
  - [FreeToken](../../wiki/entities/paper-freetoken.md) — 问题设定与系统定位。
  - [Kimi K3](../../wiki/entities/kimi-k3.md) — 典型需本地/自托管的大 MoE 权重范例。

### 2) 带宽自适应 MoE 执行（$q^\star$ policy）

- **摘录要点：** 不把个人机当「小 GPU」，而是 **GPU + CPU + 主机内存 + 互联** 的统一平台。MoE 策略含 `fused`（专家全驻 VRAM）、`offload`（专家在主机 RAM、LRU GPU 槽）、`cpu`（miss 在 CPU 算）、`hybrid`（PCIe 拉取与 CPU 计算重叠，由 `ft bench bw` 校准拆分）。`auto` 对 dense 走 fused，对 MoE 默认 offload，有带宽画像时升级 hybrid。
- **对 wiki 的映射：**
  - [FreeToken](../../wiki/entities/paper-freetoken.md) — MoE 策略与工程表。
  - [DeepSeek Harness](../../wiki/entities/deepseek-harness.md) — README 列出的 coding agent 对接对象之一。

### 3) 语义锚点缓存与弹性显存

- **摘录要点：** **Semantic anchor checkpoints** 对 recurrent state 与 KV cache 做语义级锚定，agent 在工具调用 / thinking block 等上下文编辑后避免整段重算。运行时可在 **专家缓存与 KV 显存之间动态再分配**，无需重启引擎或重载权重。辅以全层 double-buffered prefill、FTW 快速权重格式、graph-compatible 执行。
- **对 wiki 的映射：**
  - [FreeToken](../../wiki/entities/paper-freetoken.md) — agentic 工作负载读法。

### 4) 规模与硬件覆盖（论文宣称）

- **摘录要点：** 支持 **20+ MoE** 与真实 coding / tool-using agent；硬件从 **8GB 笔电 GPU** 到单卡工作站。可服务规模示例：**35B 笔电**、**284B 游戏台式**、**753B GLM-5.2 单工作站 GPU**。预构建 kernel 针对 `docs/models.md` 所列 checkpoint 调优（DeepSeek-V4-Flash、Qwen3.6/3.8、GLM-5.x、gpt-oss、Gemma-4 等；多模态族支持图像输入）。
- **对 wiki 的映射：**
  - [freetoken 仓库](../repos/freetoken.md) — 模型表与 CLI。
  - [flashml 项目页](../sites/flashml-freetoken.md)

### 5) 开源状态（截至 2026-09-15，项目页核查）

- **摘录要点：** **已开源** Apache-2.0。[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) 含 CLI（`ft serve` / `ft shell` / `ft launch`）、桌面应用下载（flashml.ai）、`uv pip install "freetoken[accel]"`。OpenAI / Anthropic 兼容 API；`ft launch` 一键接 Claude Code、Codex、dsh、OpenClaw、OpenCode 等。
- **对 wiki 的映射：**
  - [freetoken.md](../repos/freetoken.md)
  - [flashml-freetoken.md](../sites/flashml-freetoken.md)

## 当前提炼状态

- [x] arXiv 摘要与 README 核心机制已对齐摘录
- [x] 项目页 / GitHub / models.md 已交叉核查
- [x] wiki 映射：`wiki/entities/paper-freetoken.md` 新建

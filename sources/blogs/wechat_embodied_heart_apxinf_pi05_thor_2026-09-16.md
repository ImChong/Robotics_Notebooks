# 机器人不接受迟到的答案！Thor 芯片上，Pi0.5 推理被干到了 26ms，10.7 倍加速

> 来源归档（blog / 微信公众号）

- **标题：** 机器人不接受迟到的答案！Thor 芯片上，Pi0.5 推理被干到了 26ms，10.7 倍加速
- **类型：** blog
- **作者：** 具身智能之心（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/-vLQvqh_BJBtErfgS1M3jg
- **发表日期：** 2026-09-16（入库日；页面未稳定暴露 `publish_time`）
- **入库日期：** 2026-09-16
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **一句话说明：** 无问芯穹联合清华、上交推出 **APXInf**——面向机器人本体的 **VLA 端侧推理引擎**（RLinf 生态延伸）；双视角 **π₀.₅ + FP8 + Jetson AGX Thor** 实测 **278 ms → 26 ms**（相对 OpenPI 基线约 **10.7×**）；LIBERO-10 **92.2%** vs baseline **92.4%**。
- **步骤 2.5（开源核查）：** **已开源** — GitHub [`RLinf/APXinf-robo`](https://github.com/RLinf/APXinf-robo)（Apache-2.0；Rust 引擎子模块 [`infinigence/ApxInf`](https://github.com/infinigence/ApxInf)）；README 含 Quick Start、`bench_pi05.py`、OpenPI 兼容 websocket serve、`eval-libero`；权重需自备（如 `lerobot/pi05_libero_base` + OpenPI `norm_stats.json`）。

## 核心摘录（归纳，非全文）

### 问题设定

- 端侧 **π₀.₅ + OpenPI** 单次推理约 **278 ms**——对人类几乎瞬时，对闭环操纵仍偏慢；指令到动作间任何等待都可能让物理世界已变（物体移动、人手介入）。
- 移动/家庭/弱网场景不能把算力押在云端：**通信时延、可靠性、功耗、散热** 与机载 **相机/定位/规划/控制** 共享带宽。
- 通用 LLM 推理框架（Continuous Batching、Paged Attention）优化 **云端吞吐**，与机器人 **小 batch、多视角、强实时** 目标错位；VLA 缺 **专属推理引擎** → 适配成本高、抖动难控。

### APXInf 定位

| 维度 | 内容 |
|------|------|
| **出品** | 无问芯穹主导，联合清华大学、上海交通大学 |
| **生态** | [RLinf](https://github.com/RLinf/RLinf) 训练/评测基建的 **端侧推理延伸**（于超老师团队 RLinf 解决 RL 后训练；APXInf 接训练完成后的本体部署） |
| **目标** | 小 batch、多视角、强实时的 **VLA/WAM 端侧推理**；Agent 协助模型接入与后续 Workflow/Skills 沉淀 |
| **代码** | [`RLinf/APXinf-robo`](https://github.com/RLinf/APXinf-robo) |

### 性能与精度（文内 + README 对照）

| 场景 | 指标 | 说明 |
|------|------|------|
| **营销实测（文内）** | 双视角 π₀.₅ + FP8 + Thor：**278 ms → 26 ms**（~10.7×） | 相对未用 APXInf 的 OpenPI 端侧基线 |
| **官方 bench（README）** | Thor FP8 P50：**41.16 ms**（10 flow steps）；**26.32 ms**（+ onestep 剪枝） | 224×224 双视角，batch 1，CUDA Graph 稳态 |
| **LIBERO-10** | **92.2%**（500 trials）vs π₀.₅ reference **92.4%** | 文内；README Thor FP8 **461/500 = 92.2%** 一致 |

### 技术要点（归纳）

- **定制算子库：** Jetson / RTX 定向融合算子、CUTLASS、cuBLASLt；配合计算图、量化、内存与流水线优化。
- **量化：** FP32→INT8/FP8；文内强调 **速度未明显牺牲任务效果**。
- **运行时：** Rust 原生内存/并发管理 + Python API；固定执行路径、预分配，压 **时延抖动**。
- **Agent 路线：** 已有算子/接口/部署流程沉淀为 Agent 可调用 Workflow；`/model-port-workflow` skill 移植新模型（README）。
- **OpenPI 兼容：** `apxinf-robo serve` websocket + 未改 `openpi-client` 即可连。

### 与通用框架对比（文内论点）

- vLLM / sglang 等：**大 batch 吞吐** vs 机器人 **单次响应 + 抖动**。
- APXInf 受 FasterTransformer、TensorRT-LLM、llama.cpp、vLLM、sglang、FlashRT 等启发，但 **优先具身 VLA/WAM + Jetson Thor/Orin**。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [APXInf 实体](../../wiki/entities/apxinf.md) | **主沉淀页**：引擎定位、Thor/Orin 基准、OpenPI 兼容、RLinf 生态 |
| [RLinf 仓库](../../sources/repos/rlinf.md) | 训练侧基建；APXInf 为部署延伸 |
| [π₀.₅ 论文实体](../../wiki/entities/paper-pi05-open-world-vla.md) | 文内 benchmark 模型 |
| [VLA 方法页](../../wiki/methods/vla.md) | 端侧推理与部署选型 |
| [VLA 真机部署指南](../../wiki/queries/vla-deployment-guide.md) | 延迟/异步/专用引擎对照 |
| [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md) | RLinf 栈索引补 APXInf |
| [NVIDIA Jetson](../../wiki/entities/nvidia-jetson.md) | Thor/Orin 机载算力平台 |

## 当前提炼状态

- [x] 公众号正文抓取
- [x] GitHub README 开源核查（步骤 2.5）
- [x] wiki 实体与交叉链规划

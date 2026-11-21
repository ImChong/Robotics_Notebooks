---
type: entity
tags: [repo, world-model, video-generation, robbyant, physical-ai]
status: complete
updated: 2026-11-21
original:
  - ../../sources/repos/lingbot-world.md
  - ../../sources/repos/lingbot-world-v2.md
related:
  - ./lingbot-vla.md
  - ./lingbot-vla-v2.md
  - ../methods/generative-world-models.md
  - ./paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md
sources:
  - ../../sources/repos/lingbot-world-v2.md
  - ../../sources/blogs/wechat_robot_yanfa_opensource_algorithms_compendium.md
summary: "robbyant/lingbot-world：蚂蚁灵波开源世界模型主线（视频生成 + 高保真动力学）；文内 antgroup/lingbot 链接已校正。"
---

# LingBot-World

[**robbyant/lingbot-world**](https://github.com/robbyant/lingbot-world)（蚂蚁 **Robbyant / 灵波**）是 **开源世界模型** 主线仓库：从视频生成出发构建 **高保真、鲁棒动力学** 的交互式世界模拟器；后继版本见 [lingbot-world-v2](https://github.com/robbyant/lingbot-world-v2) 与论文 [Advancing Open-source World Models](./paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md)。

## 一句话定义

**灵波世界模型代码入口** — 长时序视频–动作联合建模与开源世界模拟器；与 [LingBot-VLA](./lingbot-vla.md) 操纵基础模型分工。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测 / 模拟 |
| VLA | Vision-Language-Action | 姊妹线 LingBot-VLA |
| Physical AI | Physical Artificial Intelligence | 文内「具身大模型」语境 |
| MJCF | MuJoCo XML Format | 与机器人仿真资产可组合 |

## 为什么重要

- **微信清单校正：** 原文 `antgroup/lingbot` **404**；canonical 仓为 **robbyant** org 下 `lingbot-world` / `lingbot-world-v2`。
- **长时序任务：** 文章强调「长周期连续任务预测、视频–动作联合建模」，对应世界模型而非纯 VLA 推理。
- **与 VLA 栈并列：** 同一团队 [LingBot-VLA](./lingbot-vla.md) 负责操纵；World 负责 **预测 / 仿真**。

## 核心原理

```mermaid
flowchart LR
    IN["首帧图像 + 文本提示 + 控制序列 (poses.npy / action_string)"] --> ENC["T5 文本编码 + Wan VAE 首帧条件编码"]
    IN --> CAM["Plucker 相机/动作嵌入 (AdaLN 注入)"]
    ENC --> DIT["因果块注意力 DiT (MoE / causal_fast, 维持 KV Cache)"]
    CAM --> DIT
    DIT --> SCH["FlowUniPC 多步/少步流匹配调度"]
    SCH --> DEC["Wan VAE 解码 -> 长时交互视频 MP4"]
```

图依据：[arXiv:2601.20540v1](https://arxiv.org/html/2601.20540) §3（三阶段演化与 Plücker AdaLN 动作注入）与 `robbyant/lingbot-world-v2@1895d30` `wan/image2video.py`。

**核心机制：** 基于 Wan2.2 14B 图生视频扩散基座，经三阶段演化构建世界模拟器（详见[论文索引页](./paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md)）：Stage I 通用视频先验 → Stage II 双向 MoE 中期训练（5 秒渐进到 60 秒长程一致，Plücker 嵌入与动作向量通过 AdaLN 调制 DiT）→ Stage III 块因果注意力与少步蒸馏（KV cache 支撑 16 fps、<1 s 延迟实时交互）。仓库层面，`robbyant/lingbot-world`（1.0，Apache-2.0，含 `Base Cam` / `Base Act` / `Fast`）现已停止积极维护，官方将后续更新与 2.0（14B + 1.3B，`CC BY-NC-SA 4.0`）全部迁移至 `robbyant/lingbot-world-v2`。

## 工程实践

1. 1.0 仓 `robbyant/lingbot-world`（Apache-2.0）包含 `generate.py` 与 `generate_fast.py` / `run_fast.sh`；新部署官方推荐迁移至 [lingbot-world-v2](https://github.com/robbyant/lingbot-world-v2)（[HF 集合](https://huggingface.co/collections/robbyant/lingbot-world-v2)）。
2. v2 栈依赖 `torch>=2.4`、`flash-attn` 与多卡 FSDP / Ulysses 序列并行（`--dit_fsdp --t5_fsdp --ulysses_size`）；1.3B causal-fast 变体共享 14B 的 T5/VAE 资产（`--assets_dir`）。
3. **许可差异：** 1.0 仓为 **Apache-2.0**；2.0 仓 `lingbot-world-v2` 为 **CC BY-NC-SA 4.0**（非商业共享），商业集成前务必核对版本与许可。

## 局限与风险

- **非机器人低层控制器：** 输出像素视频世界演化而非关节力矩，不能直接替代 [rsl-rl](./rsl-rl.md) 步态或操作策略。
- **org 与维护状态：** 勿使用失效的 `antgroup/lingbot` 链接；`robbyant/lingbot-world`（1.0）README 已声明不再积极维护。

## 源码运行时序图

入口：`torchrun --nproc_per_node=8 generate.py --task i2v-A14B --ckpt_dir <权重目录> --image examples/03/image.jpg --action_path examples/03 ...`（或 `bash run_fast.sh <权重目录> <帧数>`）。`--infer_mode` 默认 `causal_fast`，图以该分支为例。

```mermaid
sequenceDiagram
    participant User as 用户（torchrun）
    participant CLI as generate.py generate()
    participant Run as generate.py run_causal()
    participant Pipe as wan/image2video.py WanI2VCausal
    participant T5 as T5EncoderModel
    participant VAE as Wan2_1_VAE
    participant DiT as WanModelFast（causal_fast）
    participant Cam as utils/cam_utils.py
    participant Sch as FlowUniPCMultistepScheduler

    User->>CLI: 启动 generate.py（每卡一个进程）
    CLI->>CLI: world_size>1 时 init_process_group；ulysses_size>1 时 init_distributed_group()
    CLI->>CLI: 读取 --image；广播 base_seed
    CLI->>Run: run_causal(args, cfg, img, device, rank, mode=infer_mode)
    Run->>Pipe: WanI2VCausal(ckpt_dir, local_attn_size, sink_size, infer_mode, assets_dir)
    Pipe->>T5: 加载文本编码器
    Pipe->>VAE: 加载 VAE（_resolve_asset_path）
    Pipe->>DiT: load_dit_model(WanModelFast, fast_checkpoint)；_configure_model()（FSDP / 序列并行）
    Run->>Pipe: generate(prompt, img, action_path, chunk_size, frame_num, ...)
    Pipe->>Pipe: 按 infer_mode 分派到 _generate_causal_fast()
    Pipe->>Pipe: 读取 poses.npy，帧数裁剪为 4n+1；预处理首帧图像
    Pipe->>Cam: interpolate_camera_poses → compute_relative_poses → get_plucker_embeddings
    Pipe->>T5: 编码 prompt（结果缓存到 _t5_cache）
    Pipe->>VAE: encode(首帧 + 零帧拼接) 得到条件 y
    Pipe->>Sch: set_timesteps()，取 timesteps_index 对应的时间步
    loop 每个 chunk（chunk_size 个潜变量帧）
        loop 每个时间步
            Pipe->>DiT: model(x, t, context, c2ws_plucker_emb, kv_cache, crossattn_cache, current_start)
            DiT-->>Pipe: noise_pred
            Pipe->>Pipe: _convert_flow_pred_to_x0()
            Pipe->>Sch: add_noise(x0, 新噪声, 下一时间步)
        end
        Pipe->>DiT: 以 t=0 前向一次，更新 KV cache
    end
    Pipe->>VAE: decode(拼接后的潜变量)
    Pipe-->>Run: 视频张量
    Run-->>CLI: video
    CLI->>CLI: rank 0：save_video(mp4, fps=cfg.sample_fps)；销毁进程组
```

图依据：robbyant/lingbot-world-v2@1895d30，读过 `README.md`（Inference 段）、`run_fast.sh`（全文）、`generate.py`（`run_causal`、`generate`、`_parse_args` 中 `--infer_mode`、`--chunk_size`、`--assets_dir` 参数段）、`wan/__init__.py`、`wan/image2video.py`（`__init__` 中模型加载段、`generate` 分派段、`_generate_causal_fast` 的相机、编码、采样与 KV cache 段）。`wan/modules/model_fast.py`、`wan/utils/cam_utils.py` 的内部实现未读；`T5EncoderModel`、`Wan2_1_VAE` 未读。未运行代码（无权重与 GPU）。

## 关联页面

- [Robbyant（蚂蚁灵波）](./robbyant.md) — 发布方公司页与 LingBot 全栈
- [LingBot-VLA](./lingbot-vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [LingBot-World 论文索引](./paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md)

## 参考来源

- [sources/repos/lingbot-world-v2.md](../../sources/repos/lingbot-world-v2.md)
- [wechat_robot_yanfa_opensource_algorithms_compendium.md](../../sources/blogs/wechat_robot_yanfa_opensource_algorithms_compendium.md)
- 源码核对（2026-11-02）：robbyant/lingbot-world-v2@1895d30，读过 README.md（Inference 段）、run_fast.sh、generate.py（run_causal、generate、_parse_args 相关参数段）、wan/__init__.py、wan/image2video.py（__init__ 模型加载段、generate 分派段、_generate_causal_fast 段）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- GitHub：<https://github.com/robbyant/lingbot-world>
- 技术页：<https://technology.robbyant.com/lingbot-world-v2>

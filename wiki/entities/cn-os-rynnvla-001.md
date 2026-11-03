---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/rynnvla-001.md
summary: "阿里巴巴 开源项目 RynnVLA-001（VLA/操作模型）：把语言任务、视觉观测和机器人状态映射为动作序列，构成Rynn系列早期的通用操作基线；保留该版本有助于比较002在数据、结构和长时任务上的改动。…"
institutions:
  - alibaba
---

# RynnVLA-001

## 一句话定义

**RynnVLA-001** 是 [阿里巴巴](https://github.com/amap-cvlab) 公开的 **VLA/操作模型** 开源项目：把语言任务、视觉观测和机器人状态映射为动作序列，构成Rynn系列早期的通用操作基线；保留该版本有助于比较002在数据、结构和长时任务上的改动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第四层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 阿里巴巴 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/amap-cvlab、https://github.com/alibaba-damo-academy |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `RynnVLA-001` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python inference_lerobot.py --vla_ckpt_path <VLA 权重> --action_vae_path <ActionVAE 权重> --configs_path <yaml>`（官方示例推理脚本；其中 `img`、`wrist_img`、`state`、`lang_annotation` 在脚本里均为 `None` 占位，需由使用者接入自己的机械臂观测）。训练入口另见 `bash scripts/actionvae/action_vae.sh`、`bash scripts/lerobot/lerobot.sh`（均调用 `train.py --config_file ...`）。

```mermaid
sequenceDiagram
    participant Main as inference_lerobot.py main()
    participant Inf as inferencer/lerobot_inferencer.py LeRobotInferencer
    participant VLA as models/chameleon_model/chameleon RynnVLAForActionPrediction
    participant Tok as Chameleon 图像 VQGAN 分词 + 文本 Tokenizer
    participant AVAE as models/actionvae_model.py ActionVAE
    Main->>Main: 读取 yaml 为 opt，写入 actionvae_pretrained_path
    Main->>Inf: LeRobotInferencer(vla_ckpt_path, precision, opt, device)
    Inf->>VLA: from_pretrained(ckpt)
    Inf->>Tok: 载入 pretrained_models/Chameleon 下的 text_tokenizer.json、vqgan.yaml、vqgan.ckpt
    Inf->>AVAE: 构造并 load_state_dict(action_vae_path)，eval
    Inf->>Inf: 读 train_dataset.data_path 指向的动作与状态统计量
    Main->>Inf: step(obs, lang_annotation)
    Inf->>Inf: image_transforms 处理前视图与腕部图
    Inf->>Tok: get_condition_tokens：文本 token、quantize_images 得到两路图像 token
    Inf->>Inf: 拼 condition_tokens，末尾加 state_token_id，状态按均值方差归一化
    Inf->>VLA: generate(condition_tokens, state_embeds, GenerationConfig(max_new_tokens=1))
    VLA-->>Inf: generation_result 与 action_pred 隐变量
    Inf->>AVAE: decode_inputs(选出 action_token_id 位置的 action_pred)
    AVAE-->>Inf: 动作块，view 为 action_chunk_size x action_dim
    Inf->>Inf: 反归一化，前 5 维 cumsum 并加当前 state[:5]
    Inf-->>Main: (None, output_actions)
```

图依据：alibaba-damo-academy/RynnVLA-001@416c306，读过 README.md、inference_lerobot.py、inferencer/lerobot_inferencer.py、train.py、scripts/lerobot/lerobot.sh、scripts/actionvae/action_vae.sh；未读 RynnVLAForActionPrediction.generate 内部实现。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RynnVLA-001 源码归档](../../sources/repos/rynnvla-001.md)（<https://github.com/alibaba-damo-academy/RynnVLA-001>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：alibaba-damo-academy/RynnVLA-001@416c306，读过 README.md、inference_lerobot.py、inferencer/lerobot_inferencer.py、train.py、scripts/lerobot/lerobot.sh、scripts/actionvae/action_vae.sh；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [阿里巴巴 官方组织](https://github.com/amap-cvlab)

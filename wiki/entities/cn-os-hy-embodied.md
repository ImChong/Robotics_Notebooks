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
  - ../../sources/repos/hy-embodied.md
summary: "腾讯机器人实验室 开源项目 HY-Embodied（VLA/操作模型）：仓库汇总HY-Embodied系列模型、数据、训练和评测入口，使读者能够从统一位置追踪VLA、世界模型和跨本体版本之间的关系。…"
institutions:
  - tencent
---

# HY-Embodied

## 一句话定义

**HY-Embodied** 是 [腾讯机器人实验室](https://github.com/Tencent-Hunyuan) 公开的 **VLA/操作模型** 开源项目：仓库汇总HY-Embodied系列模型、数据、训练和评测入口，使读者能够从统一位置追踪VLA、世界模型和跨本体版本之间的关系。

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
| 机构 | 腾讯机器人实验室 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/Tencent-Hunyuan |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `HY-Embodied` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

仓库实际可运行的是推理入口（含 `Hy-Embodied-0.5/` 与 `Hy-Embodied-VLM-1.0/` 两套）；`Hy-Embodied-0.5/finetune/` 有训练代码，本图未覆盖；VLM-1.0 的 README 将 Fine-tuning Code 列为未完成。

入口：`cd Hy-Embodied-0.5 && python inference.py`（`Hy-Embodied-VLM-1.0/inference/transformers/infer_hf.py` 流程相同，另加 `trust_remote_code=True` 与 `enable_thinking` 开关）

```mermaid
sequenceDiagram
    participant U as 用户 python inference.py
    participant M as main()
    participant L as load_model_and_processor()
    participant HF as transformers (AutoProcessor / AutoModelForImageTextToText)
    participant S as single_inference() / batch_inference()
    participant G as model.generate()
    U->>M: 运行脚本
    M->>L: 加载模型 tencent/HY-Embodied-0.5
    L->>HF: AutoProcessor.from_pretrained(MODEL_PATH)
    L->>HF: AutoModelForImageTextToText.from_pretrained(bfloat16)
    L-->>M: model.to(DEVICE).eval(), processor
    M->>S: single_inference(figures/example.jpg, "Describe the image in detail.")
    S->>HF: processor.apply_chat_template(messages, enable_thinking=THINKING_MODE)
    S->>G: model.generate(max_new_tokens=1024, temperature=0.8)
    G-->>S: generated_ids
    S->>HF: processor.batch_decode(截去输入部分的 output_ids)
    S-->>M: 文本结果，print
    M->>S: batch_inference(图文 + 纯文本两条消息)
    S->>HF: 逐条 apply_chat_template，再 processor.pad(padding_side="left")
    S->>G: model.generate(**batch)
    S-->>M: processor.decode(各样本新生成 token)
```

图依据：HY-Embodied@16289e9，`Hy-Embodied-0.5/inference.py`、`Hy-Embodied-VLM-1.0/inference/transformers/infer_hf.py`。

入口：`MODEL_PATH=... bash Hy-Embodied-VLM-1.0/inference/vllm/serve.sh`，再 `python example_client.py --port 8080`

```mermaid
sequenceDiagram
    participant U as 用户
    participant SH as serve.sh
    participant V as vllm serve
    participant P as hy_embodied_vllm 插件 (pyproject.toml entry-point)
    participant C as example_client.py (OpenAI SDK)
    U->>SH: pip install -e . 后运行 serve.sh
    SH->>V: vllm serve --trust-remote-code --chat-template --reasoning-parser hunyuan_v3 --tool-call-parser hy_v3
    V->>P: 通过 vllm.general_plugins 调用 hy_embodied_vllm.register()
    P-->>V: _register_models() 注册 HYV3VLForConditionalGeneration，_register_parsers() 注册 hy_v3 / hunyuan_v3
    U->>C: python example_client.py --port 8080
    C->>V: chat.completions.create(image_url base64 + text, extra_body chat_template_kwargs.enable_thinking)
    V-->>C: message.reasoning_content 与 message.content
```

图依据：HY-Embodied@16289e9，`Hy-Embodied-VLM-1.0/inference/vllm/serve.sh`、`pyproject.toml`、`hy_embodied_vllm/__init__.py`、`example_client.py`（前 80 行）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [HY-Embodied 源码归档](../../sources/repos/hy-embodied.md)（<https://github.com/Tencent-Hunyuan/HY-Embodied>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Tencent-Hunyuan/HY-Embodied@16289e9，读过 README.md、Hy-Embodied-0.5/inference.py、Hy-Embodied-VLM-1.0/inference/transformers/infer_hf.py、Hy-Embodied-VLM-1.0/inference/vllm/{serve.sh,pyproject.toml,example_client.py(前80行),hy_embodied_vllm/__init__.py}；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [腾讯机器人实验室 官方组织](https://github.com/Tencent-Hunyuan)

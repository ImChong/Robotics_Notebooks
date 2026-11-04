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
  - ../../sources/repos/gr-mg.md
summary: "字节跳动机器人团队 开源项目 GR-MG（VLA/操作模型）：GR-MG 机器人操作生成模型官方实现…"
institutions:
  - bytedance
---

# GR-MG

## 一句话定义

**GR-MG** 是 [字节跳动机器人团队](https://github.com/ByteDance-Seed) 公开的 **VLA/操作模型** 开源项目：GR-MG 机器人操作生成模型官方实现

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
| 机构 | 字节跳动机器人团队 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/ByteDance-Seed、https://github.com/bytedance |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `GR-MG` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`bash ./evaluate/eval.sh ./policy/config/train.json`（`eval.sh` 按 epoch 列表后台启动 `python3 evaluate/eval.py`，每个 epoch 绑定一块 GPU；评测基于 CALVIN）。训练入口 `policy/main.sh`、`goal_gen/train_ip2p.sh` 本图未展开。

```mermaid
sequenceDiagram
    participant SH as eval.sh
    participant E as main()（evaluate/eval.py）
    participant IP as IP2PEvaluation（goal_gen/evaluate.py）
    participant M as CustomModel（evaluate/wrapper/model_wrapper.py）
    participant P as GR_MG（policy/model/model.py）
    participant ENV as CalvinEnvWrapperRaw（CALVIN 环境）
    SH->>E: python3 evaluate/eval.py --ckpt_dir --epoch --ip2p_ckpt_path --config_path
    E->>E: 读取 config JSON，按 epoch=(\d+) 匹配 ckpt 文件
    E->>M: CustomModel(ckpt_path, configs, device)
    M->>P: 构建 GR_MG，load_state_dict(strict=True)，eval()
    E->>ENV: make_env(dataset_path, observation_space, device_id)
    E->>IP: IP2PEvaluation(ip2p_ckpt_path)（InstructPix2Pix 目标图生成）
    E->>E: evaluate_policy(model, env, ..., ip2p_model)
    loop 每条 get_sequences() 指令序列
        E->>ENV: env.reset(robot_obs, scene_obs)
        loop 每个子任务 subtask
            E->>M: model.reset()
            loop 最多 EP_LEN 步
                opt 每 20 步（i % 20 == 0）
                    E->>IP: inference([static_rgb], [语言指令 + 进度])
                    IP-->>E: goal_image
                end
                E->>M: model.step(obs, goal_image, [lang_annotation])
                M->>P: policy.evaluate(input_dict, return_progress=True)
                P-->>M: action, progress
                M-->>E: action, progress
                E->>ENV: env.step(action)
                ENV-->>E: obs, current_info
                E->>E: task_oracle.get_task_info_for_set(...) 判定子任务是否完成
            end
        end
    end
    E->>E: print_and_save(...) 写出成功率 JSON
```

注：`CustomModel.__init__` 另外构建 MAE（`vit_base`）与 CLIP 编码器作为视觉/文本特征；`rollout` 中的图像拼接与画图（`plt.savefig`）未展开。

图依据：gr-mg@f45b072，读过 README.md、evaluate/eval.sh、evaluate/eval.py（main、evaluate_policy、rollout、evaluate_sequence）、evaluate/wrapper/model_wrapper.py（__init__ 加载、step）、goal_gen/evaluate.py（IP2PEvaluation 的初始化与 inference）；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [GR-MG 源码归档](../../sources/repos/gr-mg.md)（<https://github.com/bytedance/GR-MG>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：bytedance/GR-MG@f45b072，读过 README.md、evaluate/eval.sh、evaluate/eval.py、evaluate/wrapper/model_wrapper.py、goal_gen/evaluate.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [字节跳动机器人团队 官方组织](https://github.com/ByteDance-Seed)

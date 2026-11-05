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
  - ../../sources/repos/astranav-world.md
summary: "阿里巴巴 开源项目 AstraNav-World（世界模型）：用于前瞻控制与一致性导航的世界模型…"
institutions:
  - alibaba
---

# AstraNav-World

## 一句话定义

**AstraNav-World** 是 [阿里巴巴](https://github.com/amap-cvlab) 公开的 **世界模型** 开源项目：用于前瞻控制与一致性导航的世界模型

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
| 类别 | 世界模型 |
| 官方组织 | https://github.com/amap-cvlab、https://github.com/alibaba-damo-academy |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `AstraNav-World` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 源码运行时序图

入口：`cd infer_ovon && bash eval_ovon.sh`（内部 `python -u run_infer_ovon.py --exp-config ovon/configs/ovon_pano.yaml --model-path ... --result-path ...`；R2R/RxR 为 `infer_r2r_rxr/` 下的 `eval_r2r.sh`、`eval_rxr.sh`，本图只画 OVON；未读 R2R/RxR 代码）。`PREDICT_FUTURE_FRAMES=true` 时才启用 Wan 未来帧分支，默认关闭。

```mermaid
sequenceDiagram
    participant SH as eval_ovon.sh
    participant Run as run_infer_ovon.py（main / run_exp）
    participant Ev as agent/waypoint_agent_ovon.py（evaluate_agent_ovon）
    participant Ag as Waypoint_Agent
    participant Q as QwenModel（Qwen VL 主干）
    participant AF as ActionFormer（wan/future_frames.build_navigation_model）
    participant Wan as FutureFramePredictor（Wan2.2，可选）
    participant Env as habitat.Env（OVON）

    SH->>Run: python run_infer_ovon.py（CUDA_VISIBLE_DEVICES，split-num / split-id）
    Run->>Run: habitat.get_config + make_dataset，dataset.get_splits
    Run->>Ev: evaluate_agent_ovon(config, dataset_split, model_path, result_path)
    Ev->>Ag: Waypoint_Agent(model_path)
    Ag->>Q: resolve_checkpoint_paths → QwenModel(qwen_path)
    Ag->>AF: build_navigation_model(action_path, predict_future_frames, wan_model_path)
    loop 每个 episode
        Ev->>Env: env.reset()，agent.reset()，随机选 obj_goal_template 指令
        loop 直到 env.episode_over
            Ev->>Ag: act(obs 含 left/front/right 图像、pose、instruction)
            Ag->>Ag: add_frame，generate_infer_prompt（历史图像 + 局部 pose）
            Ag->>Q: qwen_infer(messages, return_vlm_output=True)
            Q-->>Ag: visual_language_emb
            Ag->>AF: action_former(visual_language_emb)
            AF-->>Ag: 航点 wp_pred、arrive、sin/cos 角度
            opt predict_future_frames
                Ag->>Wan: predict(frames, visual_language_emb)，save_future_video
            end
            Ag-->>Ev: 航点 / arrive_pred / recover_angle
            Ev->>Env: env.step(GO_TOWARD_POINT 或 stop)
        end
        Ev->>Ev: env.get_metrics → write_result（success / spl / distance_to_goal）
    end
```

图依据：amap-cvlab/AstraNav-World@734dc04，读过 README.md、infer_ovon/eval_ovon.sh、infer_ovon/run_infer_ovon.py、infer_ovon/agent/waypoint_agent_ovon.py、world/wan/future_frames.py（build_navigation_model、predict）、world/wan/checkpoint.py（函数列表）。

## 参考来源

- [AstraNav-World 源码归档](../../sources/repos/astranav-world.md)（<https://github.com/amap-cvlab/AstraNav-World>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：amap-cvlab/AstraNav-World@734dc04，读过 README.md、infer_ovon/{eval_ovon.sh,run_infer_ovon.py,agent/waypoint_agent_ovon.py}、world/wan/{future_frames.py,checkpoint.py}；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [阿里巴巴 官方组织](https://github.com/amap-cvlab)

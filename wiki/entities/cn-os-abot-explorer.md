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
  - ../../sources/repos/abot-explorer.md
summary: "阿里巴巴 开源项目 ABot-Explorer（具身Agent/规划）：VLM 驱动、支持 3DGS 与 Habitat 的自主探索智能体…"
institutions:
  - alibaba
---

# ABot-Explorer

## 一句话定义

**ABot-Explorer** 是 [阿里巴巴](https://github.com/amap-cvlab) 公开的 **具身Agent/规划** 开源项目：VLM 驱动、支持 3DGS 与 Habitat 的自主探索智能体

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
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/amap-cvlab、https://github.com/alibaba-damo-academy |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `ABot-Explorer` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`cd explore_agent && CUDA_VISIBLE_DEVICES=0 python scripts/infer_close_3dgs_singal.py -c config/infer_interiorgs.yaml -s {SCENE_ID}`（3DGS 路径；需先在 `gaussian_splatting_sim/` 下运行 `python render_sim.py --load_ply_root <gs目录> --gpus 0` 启动渲染服务。Habitat 路径入口为 `python scripts/infer_close_habitat.py -c config/infer_habitat.yaml`，本图未展开。）

```mermaid
sequenceDiagram
    participant U as 用户终端
    participant M as scripts/infer_close_3dgs_singal.py<br/>main 与 process_graph_data
    participant C as utils/params.py<br/>读取 -c 配置
    participant Q as modules/model_qwen2_5.py<br/>QwenModel2_5
    participant T as modules/topograph.py<br/>TopoGraph
    participant S as utils/scene.py<br/>load_map_data
    participant R as utils/render.py<br/>render_gs 与 get_render_image
    participant G as gaussian_splatting_sim/render_sim.py<br/>Flask /render_gs 端口 7001
    U->>M: python infer_close_3dgs_singal.py -c config/infer_interiorgs.yaml -s SCENE
    M->>C: 模块导入时 yaml.safe_load 得到 render_url map_dir result_dir 等
    M->>Q: QwenModel2_5(model_path) 加载 Qwen2.5-VL 处理器与模型
    M->>M: makedirs 后 shutil.rmtree 清理 result_dir/SCENE
    M->>T: TopoGraph(merge_dist, vis_dist, merge_dir, step_dist)
    loop repeat_num 次轨迹
        M->>S: load_map_data(map_dir/SCENE) 读取 link_map structure occ_map 等
        S-->>M: map_dict
        M->>M: process_graph_data 随机起点 get_random_start
        M->>R: render_gs(cur_pos, cur_dir, camera_height, 保存路径) 四个视角
        loop front left right back 四个视角
            R->>G: POST render_url 携带 cam_extrinsics 与 cam_intrinsics
            G->>G: render_set_one_image 调用 pgsr 渲染为 JPEG
            G-->>R: image/jpeg
        end
        R-->>M: 四张视角图
        M->>Q: qwen_data_pack 加 generate_scene_question 后 qwen_infer
        Q-->>M: 场景文本
        M->>M: parse_scene_answer 解析为 scene_dict
        M->>T: add_pose 与 add_scene
        alt decision_mode 为 normal
            M->>T: get_target_node
        else decision_mode 为 infer
            M->>T: get_local_graph 后 generate_graph_question
            M->>Q: qwen_infer 返回 target_id
        end
        M->>T: get_path(cur_id, target_id) 得到 next_id
        loop 直到全部探索或达到 max_step
            M->>R: render_gs 沿路径按 step_dist 与 step_angle 前进
            M->>T: add_pose get_near_node reset_cur_node
            opt 到达目标节点
                M->>Q: 再次场景问答后 add_scene
            end
            M->>M: metric_interiorgs 计算覆盖率与路径长度
        end
        M->>T: build_clean_graph
        M-->>U: 写出 graph.json 与 result.json
    end
```

图依据：amap-cvlab/ABot-Explorer@6abe23a，读过 README.md、explore_agent/scripts/infer_close_3dgs_singal.py、explore_agent/scripts/infer_close_habitat.py、explore_agent/utils/params.py、explore_agent/utils/render.py、explore_agent/modules/model_qwen2_5.py、explore_agent/config/infer_interiorgs.yaml、explore_agent/bash/eval_interiorgs.sh、gaussian_splatting_sim/render_sim.py（路由与启动段）；函数位置另经 grep 确认（topograph.py、scene.py、prompt.py、geometry.py、visualize.py）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [ABot-Explorer 源码归档](../../sources/repos/abot-explorer.md)（<https://github.com/amap-cvlab/ABot-Explorer>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：amap-cvlab/ABot-Explorer@6abe23a，读过 README.md、explore_agent/scripts/infer_close_3dgs_singal.py、explore_agent/scripts/infer_close_habitat.py、explore_agent/utils/params.py、explore_agent/utils/render.py、explore_agent/modules/model_qwen2_5.py、explore_agent/config/infer_interiorgs.yaml、gaussian_splatting_sim/render_sim.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [阿里巴巴 官方组织](https://github.com/amap-cvlab)

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
  - ../../sources/repos/robomind-sim.md
summary: "北京人形机器人创新中心 开源项目 RoboMIND-Sim（仿真环境）：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证…"
institutions:
  - x-humanoid
---

# RoboMIND-Sim

## 一句话定义

**RoboMIND-Sim** 是 [北京人形机器人创新中心](https://github.com/Open-X-Humanoid) 公开的 **仿真环境** 开源项目：官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 北京人形机器人创新中心 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/Open-X-Humanoid |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `RoboMIND-Sim` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python3 RoboMIND-Sim/tools/policy_infer.py`（策略侧，仅 ZMQ 收发骨架，模型推理需用户自行实现）；再 `~/isaacsim_5.1/python.sh RoboMIND-Sim/benchmark.py --task TienKung_task_01 --loop 3 --timeout 300`（批量评测），单次可直接运行 `tasks/run_task.py`。

```mermaid
sequenceDiagram
    participant PI as tools/policy_infer.py (ZMQ 收发)
    participant B as benchmark.py (BenchmarkRunner)
    participant TM as tasks/task_manager.py (TaskManager)
    participant RT as tasks/run_task.py (子进程, Isaac Sim python.sh)
    participant TK as TienKung_task_base.py / TienKung_task_01.py
    participant R as robots/TienKung2_Inspire.py

    PI->>PI: ZmqReceiver(5556) 订阅 + ZmqPublisher(5557) 绑定，等待消息
    B->>TM: TaskManager(--task) ，loop 次 run_task(timeout)
    TM->>TM: select_random_usd_path(task toml 的 environment.usd_path)
    TM->>RT: subprocess.Popen(isaac python, run_task.py --task --status-file --usd-path)
    RT->>RT: SimulationApp(config) 启动后 import robots / tasks.TienKung_tasks
    RT->>TK: Registry.create(task) → 打开 USD 场景，Registry.create("TienKung2_Inspire")
    TK->>R: 构造机器人与 camera_head(SimCam)
    RT->>TK: start()
    TK-->>PI: ZMQ PUB 5556: topic test，随后 topic start(b"TienKung")
    PI-->>TK: ZMQ PUB 5557: topic test 回应
    RT->>TK: play(num_steps)：init_play 预热、注册 physics callbacks、to_arm_init_pose → to_home_pose
    loop 每个仿真步 one_step
        TK->>R: collect_data_callback(每 4 步) pub_l_r_joints + get_rgb
        TK-->>PI: topic obs(关节+head 相机图像)
        Note over PI: 当前代码仅显示图像并打印，推理已被注释，不发送 action
        PI-->>TK: topic action(若用户实现)
        TK->>R: update_joint_callback → robot.update_command(data)
        TK->>TK: check_success_callback(task_checker.check_relative_position)
    end
    TK->>TK: write_task_status(success/score) 写入 status-file；stop() 发 topic reset
    TM->>TM: 轮询等待进程结束或超时，读取 status-file 的 success
    TM-->>B: 返回成功与否；B 统计 Success rate，stop_task_process
```

图依据：Open-X-Humanoid/RoboMIND-Sim@0415190，读过 `README.md`、`benchmark.py`、`tools/policy_infer.py`、`tasks/run_task.py`、`tasks/task_manager.py`（`start_task_process`、`run_task`）、`tasks/task_base.py`、`tasks/TienKung_tasks/TienKung_task_base.py`、`TienKung_task_01.py`、`common/utils/zmq_utils.py`、`robots/TienKung2_Inspire.py`（仅类/函数签名）。`task_02~04`、`robots/arm_base.py`、`common/utils/task_check.py` 的实现细节未读；`robot.update_command` / `pub_l_r_joints` 的内部关节处理未展开。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [RoboMIND-Sim 源码归档](../../sources/repos/robomind-sim.md)（<https://github.com/Open-X-Humanoid/RoboMIND-Sim>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Open-X-Humanoid/RoboMIND-Sim@0415190，读过 README.md、benchmark.py、tools/policy_infer.py、tasks/run_task.py、tasks/task_manager.py、tasks/task_base.py、tasks/TienKung_tasks/TienKung_task_base.py、TienKung_task_01.py、common/utils/zmq_utils.py、robots/TienKung2_Inspire.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [北京人形机器人创新中心 官方组织](https://github.com/Open-X-Humanoid)

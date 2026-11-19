# CloudRobo-WM｜XPolicyLab RoboTwin 2.0 适配代码

- **类型：** 公开源代码与作者提供的评测记录
- **源码：** https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/CloudRobo_WM
- **说明文档：** https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/CloudRobo_WM/README.md
- **核对范围：** 已完整阅读该 policy README 与 eval.sh、deploy.py、setup_eval_policy_server.sh、train.sh，检查 model.py 的输入适配与构造代码。尚未对大权重进行下载、模型执行或独立复现。
- **通读：** 已完成
- **入库日期：** 2026-11-19

## 适配目标与输入输出

- 评测：RoboTwin 2.0 / arx_x5 / joint 动作空间，单服务端一个仿真环境，不支持 eval_batch。
- 观测：cam_head、cam_left_wrist、cam_right_wrist 三路 uint8 RGB；双臂六关节 + 左右夹爪，共 Joint14。
- checkpoint：weights/cloudrobo_wm_bf16.safetensors，及 third_party 视觉/视频依赖。
- 推理配置：num_inference_steps=4、replan_steps=32、inference_seed=42；这三项为适配器默认配置，非 WM1.2 官方延迟保证。

## 已检查的运行入口

- 安装：bash install.sh --python /path/to/python；先行 bash smoke_test.sh --static。
- 下载：bash download_checkpoint.sh --output-dir /models/cloudrobo-wm；可用 --verify-only 和 --source-dir。
- 评测：bash eval.sh RoboTwin adjust_bottle /models/cloudrobo-wm arx_x5 joint 0 0 1 auto robotwin。
- eval.sh 为 server 分配端口，启动 setup_eval_policy_server.sh，等待就绪后启动 setup_eval_env_client.sh；setup server 再调用 XPolicyLab/setup_policy_server.py。
- deploy.py 的 eval_one_episode 循环调用 model_client.reset、update_obs、get_action，将动作逐步发送至 TASK_ENV.take_action，达到 episode 终点则退出。
- train.sh 转发至 tools/train_entry.py；训练环境及数据前处理须参阅 posttrain/README.md。此处是 XPolicyLab 的适配实现，不代表 WM1.2 全量预训练源码。

## 结果的证据范围

README 自报 50 任务 × 每档 100 次仿真：Clean/seen 4764/5000（95.28%）；Random/unseen 4134/5000（82.68%），等权均值 88.98%；提供 results、evaluation/evidence 目录、manifest 和日志指针。尚未在当前环境重跑，**此处数值是该适配器公开评测结果**，不自动视为 WM1.2 的独立版本实测成绩。

## 权重身份

README 固定给出 HF 资产 CloudRobo-WM-1.0-RoboTwin-20261111 的后训练权重和初始化权重，强调二者不能混用。关于预训练阶段完整权重、WM1.2 正式发布包及使用许可仍需另外核实。

## 关联

- [WM-1.0 权重归档](../models/cloudrobo-wm-1-0-robotwin-20261111.md)
- [WM1.2 官方发布记录](../sites/cloudrobo-wm12-2026.md)
- [CloudRobo-WM 家族实体](../../wiki/entities/cloudrobo-wm.md)

# TidyBot 官方代码与运行入口

- **仓库：** https://github.com/jimmyyhwu/tidybot
- **固定提交：** c060278de506699b46ff7361357109cf9dcd5654
- **论文：** [v2 归档](../papers/agentic_2305_05658.md)；[项目页核查](../sites/agentic-tidybot-2305-05658.md)。
- **核对范围：** README 的服务端/机器人端结构、Fully Autonomous Mode；server/demo.py 的初始化、update_command、run，server/controller_client.py、server/controller_server.py 的队列转发；robot/controller.py 的 handle_object_detection、run、build_base_command、build_arm_command。相关入口静态阅读，未全仓审计或运行硬件。
- **通读：** 未完成

## 已核实的调用与边界

工作站启动 object_detector_server.py 和 demo.py；机器人启动 controller.py，底盘驱动和停止控制另行准备。demo 默认加载 scenarios/test.yml 与 preferences/test.yml，偏好不是控制周期中实时调用语言模型产生的。顶视图送检测服务，update_command 找最近目标；ControllerClient.execute_command 先向 controller_server 发送命令，服务端通过队列和机器人状态轮询交接；机器人控制器先移动底盘，再执行机械臂原语。第一视角分类由 handle_object_detection 线程请求 use_clip 服务，类别随控制状态回传给 demo。接着由 YAML 中 placements/primitives 选择容器与原语。

原文 Figure 2 是概念流程，发布代码的分类请求在抓取附近异步发起，不应把概念图当成逐函数执行顺序。默认演示模式需逐命令确认；自主模式显式切换。相机标定、容器位置、机器人驱动与历史模型依赖均需实际配置。

## 对 wiki 的映射

[独立 TidyBot 节点及运行时序图](../../wiki/entities/paper-tidybot-2305-05658.md)。TidyBot2 属于不同论文，不合并身份。

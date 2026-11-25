# msgq：共享内存进程间消息队列 / msgq IPC — 官方源码归档

- **资料类型：** 开源代码仓库，官方一手资料
- **项目：** https://github.com/commaai/msgq
- **仓库分支：** master
- **核对提交：** [42554d0a6f1cb2e87238c994760078be36cfdb27](https://github.com/commaai/msgq/commit/42554d0a6f1cb2e87238c994760078be36cfdb27)
- **核对日期：** 2026-11-22
- **代码：** https://github.com/commaai/msgq（已开源；未本地运行硬件）
- **核对范围：** README.md、msgq/__init__.py、msgq/examples/publisher.py、msgq/examples/subscriber.py
- **通读：** 已完成
- **对应节点：** [wiki/entities/commaai-msgq.md](../../wiki/entities/commaai-msgq.md)

## 来源所支持的结论

msgq 是 comma.ai 的进程间发布订阅消息库，一个命名端点对应一个发布者、多个订阅者；它是 openpilot cereal 的底层消息传输后端。

调用 msgq.pub_sock(endpoint) 创建发送端，msgq.sub_sock(endpoint, timeout=..., conflate=...) 创建订阅端；发布方 send(bytes)，接收方 receive()，可借助 Poller 观察多个端点。README 描述队列由元数据及 ring buffer 组成；每条消息含 8 字节长度前缀。写入覆盖滞后的读者时可将其标为失效，因此慢订阅者可能丢消息；conflate 模式读取最新可用消息。openpilot 另通过 cereal/Cap’n Proto 定义数据模式，不能混称 msgq 自带这些消息结构。

## 实际源码与运行入口

- **核对路径：** msgq/__init__.py、msgq/examples/publisher.py、msgq/examples/subscriber.py
- **官方快速开始／复现范围：** 可复核的官方示例：安装 msgq-ipc 后分别运行 python -m msgq.examples.publisher --endpoint demo 与 python -m msgq.examples.subscriber --endpoint demo；用同名 endpoint 接收。测试命令是仓库 ./test.sh。对于控制链路需评估队列溢出、读端滞后、数据新鲜度及重放需求。

## 开源范围与风险边界

已通读官方 README 含 Under the hood 读写规则，并核对 Python API、发布端及订阅端例程；未复跑延迟基准。

接口不等同于可靠的持久消息队列：消息按环形缓存容量覆盖，慢订阅者可能漏收；README 的基准图不能直接当作所有机器人平台的实时延迟保证。

## 上下游关系

- [commaai/openpilot](https://github.com/commaai/openpilot)
- [commaai/panda](https://github.com/commaai/panda)
- [commaai/opendbc](https://github.com/commaai/opendbc)

> 本文件只保留核对范围及关键事实，不镜像粘贴官方 README 或受版权保护的整段源码；具体实现以锁定提交为准。

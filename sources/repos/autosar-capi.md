# AUTOSAR/capi — 自适应平台通用实现

> 来源归档

- **标题：** AUTOSAR CAPI
- **类型：** repo
- **机构：** AUTOSAR；初版基线由 iSOFT 贡献
- **链接：** https://github.com/AUTOSAR/capi
- **项目页：** https://www.autosar.org/capi
- **版本：** CAPI 1.0（SDK 构建版本参数 2608）；源码核对 main 快照 112916a45757acefad9388ebba1c811cecb9c970
- **许可：** Community Source；[LICENSE](https://github.com/AUTOSAR/capi/blob/main/LICENSE) 限定用途，商业利用需授权。
- **核对范围：** README.rst、LICENSE、docs/building_and_running_CAPI.rst、docs/tutorials/demo.rst、demo/docker-compose.yml、samples/helloworld-cm/client/src/main.cpp 与 server/src/main.cpp 全文；范围为发布基线和通信 Demo，不声称逐行审计全平台。
- **通读：** 已完成

## 关键入口

| 入口 | 用途 |
|---|---|
| [README](https://github.com/AUTOSAR/capi/blob/main/README.rst) | 版本、目录、授权与复现步骤 |
| [构建运行指南](https://github.com/AUTOSAR/capi/blob/main/docs/building_and_running_CAPI.rst) | SDK 编译、打包、安装与配置 |
| [Demo 指南](https://github.com/AUTOSAR/capi/blob/main/docs/tutorials/demo.rst) | 两容器 SOME/IP 通信 |
| [client/main.cpp](https://github.com/AUTOSAR/capi/blob/main/samples/helloworld-cm/client/src/main.cpp) | 服务发现、Proxy、EchoMethod 请求 |
| [server/main.cpp](https://github.com/AUTOSAR/capi/blob/main/samples/helloworld-cm/server/src/main.cpp) | Skeleton、服务提供、事件、回显 |
| [Compose](https://github.com/AUTOSAR/capi/blob/main/demo/docker-compose.yml) | 第二容器与共享网络 |
| `isoft/sdk-utils/bin/sdk-utils.py` | 官方 SDK 构建入口 |
| `isoft/ara-gen` | 代码生成器目录 |

## 对 wiki 的映射

- [AUTOSAR CAPI 1.0](../../wiki/entities/autosar-capi.md)
- [官方站点归档](../sites/autosar-capi.md)

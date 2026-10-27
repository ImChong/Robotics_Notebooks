# Netdata

> 来源归档：根据截图中的品牌与标语识别，并核对官方 GitHub 仓库及 README。

- **标题：** Netdata
- **类型：** repo / 基础设施可观测性与实时监控
- **代码：** <https://github.com/netdata/netdata>
- **官方网站：** <https://www.netdata.cloud/>
- **文档：** <https://learn.netdata.cloud/>
- **主要语言：** Go
- **许可：** Netdata Agent 仓库标注 GPL-3.0；Netdata UI 与 Netdata Cloud 有独立许可说明，使用或再分发前应查看各自许可文件。
- **入库日期：** 2026-10-27
- **一句话说明：** Netdata 是面向服务器、容器、应用和云基础设施的实时可观测性平台，提供指标采集、仪表盘、告警与异常检测能力。

## 一手资料核查

- 官方 README 将项目定位为全栈可观测性平台，并介绍 Netdata Agent 的指标采集、存储、机器学习异常检测、告警、流式传输与查询能力。
- README 提到按秒采集和实时仪表盘，也列出系统资源、硬件传感器、容器、虚拟机、数据库及其他应用集成。
- 官方说明区分 Netdata Agent、Netdata UI 和 Netdata Cloud：Agent 为开源核心；UI 与 Cloud 的许可及功能边界分别说明。Netdata Cloud 为可选服务，部署前应按自身数据与访问需求核对配置。
- 对机器人研发而言，Netdata 可作为边缘计算机、仿真服务器及配套服务的主机监控候选；具体采集器覆盖、资源开销和告警策略需在目标硬件与工作负载上验证。它本身不是机器人控制框架。

## 主要链接

- [官方仓库 README](https://github.com/netdata/netdata)
- [官方文档](https://learn.netdata.cloud/)
- [官方主页](https://www.netdata.cloud/)
- [仓库许可文件](https://github.com/netdata/netdata/blob/master/LICENSE)

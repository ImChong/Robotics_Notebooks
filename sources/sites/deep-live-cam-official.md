# Deep-Live-Cam — 官方网站与版本边界

> 官方项目页原文归档（按官方网页核验后提炼，非转载）

- **标题：** DeepLiveCam — Real-time face swap and AI video tools
- **类型：** site / official product website
- **官方入口：** <https://deeplivecam.net/>
- **代码：** <https://github.com/hacksider/Deep-Live-Cam>
- **发布记录：** <https://github.com/hacksider/Deep-Live-Cam/releases>
- **核对日期：** 2026-11-08
- **核对范围：** 官网首页的产品介绍、硬件要求、Lite / Ultimate 订阅及云服务说明；GitHub Releases 的 2.6d、2.7、2.8 Ultimate 公告；对照官方 GitHub README 现存 2.1.6 标题。
- **通读：** 已完成
- **开源状态：** **混合发布** — GitHub 公开 Python 主分支代码；官网另行提供订阅制 Lite / Ultimate 预构建版本和限定功能，不等同于 GitHub 完整公开。

## 一手页面核验

1. 官网明确列有“OPEN-SOURCE ROOTS”，指向 \`hacksider/Deep-Live-Cam\`，同时设有 **Find your plan / subscription** 入口。故“免费开源”只适用于可自行部署的公开代码，不应扩展为“所有版本都免订阅”。
2. 官网对 **NVIDIA Windows** 建议 RTX 3060 或更新、显存 6 GB 以上，对 **AMD Windows** 建议显存 6 GB 以上；这属于**官网订阅产品建议配置**，并不是 GitHub 开源版在全部条件下保证实时帧率的基准。
3. 官网 FAQ 明言：**本地人脸替换在用户机器执行**；**可选的云 AI 工作流**（例如 Decart）使用外部服务并要求单独 API key。“全部离线”不可泛化到所有版本。
4. GitHub Releases 中 **2.6d** 公告明确引入 Virtual Camera 与 OBS 工作流；**2.7 / 2.8 Ultimate** 的多项新能力与优化明确标注需经官网获取。不能不经代码核查便断言主分支具有全部 Ultimate 功能。
5. 官网宣传的速度、硬件要求与视觉品质未附跨平台公开复现实验表；不能由营销演示推导“任何转头/嘴型均完全无误”。

## 版本划分（2026-11-08 快照）

| 发行渠道 | 可核实事实 | 选型含义 |
|---------|-----------|---------|
| GitHub \`main\` | README 标题为 Deep-Live-Cam 2.1.6；公开 Python / PySide6 / ONNX Runtime 代码 | 可审计、可自行安装与测试 |
| 官方 Release 2.6d | Release 文本宣布“Virtual Camera support ... via OBS” | 是官方**发布功能声明**，非 GitHub \`main\` 代码等价证明 |
| 官网 Lite / Ultimate | 官网明确提供订阅计划；2.8 Ultimate 早期访问 | 单独核对付费权益和软件构建 |
| 可选云 AI 集成 | 官方网站 FAQ 表明需使用外部服务/API key | **不满足纯本地** 数据路径 |

## 对 wiki 的映射

- [官方仓库源码归档](../repos/deep-live-cam.md)
- [Deep-Live-Cam 实体节点](../../wiki/entities/deep-live-cam.md)

## 原文链接

- <https://deeplivecam.net/>
- <https://github.com/hacksider/Deep-Live-Cam/releases>
- <https://github.com/hacksider/Deep-Live-Cam/blob/main/README.md>

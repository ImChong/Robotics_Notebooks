# 西湖机器人官网：GAE 下载页与「具身大模型」模块

- **URL：** <https://www.wlrobo.com/GAEDownload>（GAE 下载）；<https://www.wlrobo.com/module1>（具身大模型，官网根路径重定向至此）
- **机构：** 西湖机器人科技（杭州）有限公司（Westlake Robotics）
- **核查日期：** 2026-10-10（Playwright 渲染 innerText + 前端 bundle 中的下载接口；下载接口返回的 OSS 地址为 1 小时签名链接，下文只记录不含签名的路径）
- **关联来源：** [GAE 项目页](gae-general-action-expert.md)；[GAE 论文归档](../papers/gae_general_action_expert_arxiv_2609_34233.md)
- **关联知识页：** [GAE 全身遥操作](../../wiki/entities/paper-gae-general-action-expert.md)；[西湖机器人](../../wiki/entities/westlake-robotics.md)；[TITAN O1](../../wiki/entities/westlake-titan-o1.md)
- **代码 / 权重 / 数据：** 下载页提供**闭源商用二进制安装包**（Windows 客户端 + 机器人端安装包），不提供算法源码、训练代码、模型权重文件或训练数据；使用需购买并配套 USB 加密狗与密钥。

## 具身大模型模块（module1）原文要点

- 标题：「全身运动操作一体 通用具身基础大模型」。
- **大脑**：「LM-VLM · 多模态运动操作模型」，演示视频配文「毛衣收纳，衣角通通塞进去」。页面无论文、技术报告、代码或参数说明链接。
- **通用小脑**：「GAE」，「身外化身 如影随形」，「全球首个实现一脑多形的“准AGI”控制系统」（自报宣传语）。
- 关于我们页（module4）时间线自报：2025-03-10「发布国际上第一个人形机器人全身运动大模型」；**2025-09-29「GAE身外化身系统发布」**；并称公司为「全球唯一通用大小脑双pretrain架构」「全球唯一的通用行为专家大模型GAE」（自报）。

## GAE 下载页（GAEDownload）

页面两组按钮：「GAE身外化身系统」（国内版，`dist_type=0`）与「GAE Emanation」（海外版，`dist_type=1`）；点击后弹窗经 `/api/oss_access/gae_release_dist/*` 接口取版本元数据与签名下载链接。2026-10-10 接口返回：

| 项 | 内容 |
|----|------|
| 最新版本 | `latest_release_version: V1.2.0`；客户端与机器人端兼容表均只列 V1.2.0 |
| Windows 客户端 | 国内 `gae/V1.2.0/cn/WLRobot_WebService_Setup_v1.2.0_v11.exe`（约 829 MB，OSS Last-Modified 2026-06-15）；海外 `gae/V1.2.0/over_sea/WLRobot_WebService_Setup_v1.2.0_Oversea_v1.exe`（约 829 MB，2026-06-10） |
| 机器人端安装包 | `robot_type: unitree`；国内 3 个变体 Ubuntu 20.04 / CUDA 11 / TensorRT 8、Ubuntu 22.04 / CUDA 12 / TensorRT 10、Ubuntu 22.04 / CUDA 12 / TensorRT 8，文件名带 `v1.2.0_cn-beta.8`；海外 2 个变体（缺 22.04 + TRT8），文件名 `v1.2.0_en`；每个约 1.26–1.27 GB，OSS Last-Modified 2026-06-10 至 2026-06-22 |
| 安装包形态 | 自解压 `.run`：要求 root（自动 `sudo`），校验内嵌 SHA-256 后解出 `dist/deploy.sh` 与内层 `WLR_GAE-<os>_<cuda>_<trt>.tar.gz` 并执行部署脚本；内层归档未解开核查 |
| 环境检测脚本 | `gae/V1.2.0/detect_jetson_env.sh`（3.9 KB，头注释 `2025-11-14 v1.0.0`）：读取 Jetson 型号、Ubuntu、JetPack、CUDA、cuDNN、TensorRT 版本，输出应选的安装包名；注释写「适用于 Jetson Orin / Xavier / Nano」 |
| 许可 | 下载页未给出许可证文本；无开源许可 |
| 产品文档 | 「GAE身外化身系统使用说明书」→ 飞书文档《GAE操控平台产品使用手册》（页面显示「8月24日修改」，未显示年份）；「GAE Emanation Instruction Manual」→ 飞书《GAE Control Platform Product User Manual》（「6月8日修改」），内含安装与诺亦腾动捕操控视频 |

## 产品使用手册要点（飞书，自报功能）

- 定位：GAE（General Action Expert）身外化身平台是「面向具身机器人大模型操控系统」，提供单机遥操（无线、局域网）、远程遥操（仅无线网络）与群控（局域网 / 无线，最多 10 台）。
- 输入：键鼠「动作引擎控制」（WASD + 鼠标转向）、动捕控制（列出诺亦腾、青瞳视觉、凌云光、虚拟动力、魔迅、动见、Xsens、度量、Vicon）、PICO VR、API 控制（专业版），以及上下半身分离的「混合控制」。
- 机器人端模型两种：「跃动模型」（宣传语「跑起来，跳起来」，不支持灵巧手）与「灵动模型」（更低延迟，支持灵巧手）。灵巧手实时同步 / 离散状态映射为专业版功能。
- 动作库与录制：动作广场（舞蹈、武术、日常）、3 秒–5 分钟动作录制，可导出 `动作名+机型后缀.json`（导出为专业版）。
- 数据采集：真机遥操轨迹与视觉数据清洗、时间戳对齐并序列化为 mpds 格式，供模仿学习 / 强化学习。
- 鉴权：USB 加密狗读取硬件 PID 与功能授权位；国内版支持手机号或加密狗登录，海外版仅加密狗。AR（安卓「GAE Vision」）/ VR 连接标注「此功能近期发布」。
- API 文档指向 GitHub `westlakerobotics/GAEApp-API`。

## GitHub：westlakerobotics/GAEApp-API

- **URL：** <https://github.com/westlakerobotics/GAEApp-API>（2026-10-10 `git clone` 核查，HEAD `cdcfd67`）
- 内容：GAE 客户端 WebSocket / HTTP 接口说明（PDF「GAEApp-API(WebSocket版本) V1.0.0」、Apifox 导出 HTML）、骨骼 FBX、Python 示例（`SkeletonStreamControl` 以 60 Hz 向 `ws://localhost:9003/WLRSendBoneCMD` 发送骨骼帧；`RobotJointStreamControl` 发送关节帧；另有机器人状态 / 错误码订阅接口）。
- README：「购买并下载西湖机器人GAE软件，西湖机器人会提供加密狗与密钥」，在客户端「API控制」模式下使用。
- 提交时间：首个提交 2026-03-02，最近提交 2026-08-10；仓库根目录无 LICENSE 文件。
- 结论：这是**闭源客户端的对接接口与示例**，不是 GAE 训练 / 推理算法代码。

## 对 wiki 的映射

- GAE 开放状态由「未列可获取实现」更新为「**商用闭源二进制可下载，算法代码 / 权重 / 数据未开源**」→ [GAE 实体页](../../wiki/entities/paper-gae-general-action-expert.md)。
- 产品时间线：2025-09-29 首次发布（官网与项目页自报）早于 2026-09-28 论文 v1。
- 「大脑 LM-VLM」在官网仅有名称与演示配文；截至 2026-10-10 未检索到论文、技术报告或代码，不单独建页。

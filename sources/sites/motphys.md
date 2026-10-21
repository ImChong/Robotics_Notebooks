# Motphys（谋先飞）：官网首页、关于、研究与版本日志核查

- **类型：** 官方站点索引（首页 + 关于 + 研究 + 版本日志）
- **名称：** Motphys（法人主体：北京谋先飞技术有限公司，页脚 © 2026）
- **官网：** <https://www.motphys.com/zh/>
- **子页：** [关于](https://www.motphys.com/zh/about/)、[研究](https://www.motphys.com/zh/research/)、[版本日志](https://www.motphys.com/zh/release-notes/)
- **核查日期：** 2026-10-17
- **方法：** 读取上述四个页面正文；数字与日期照抄页面，页面未给出者不补。
- **一句话说明：** 自研实时物理引擎起家，向上做训练框架与数据引擎的具身仿真基础设施公司；不训练机器人基础模型。

## 公司信息（官网自述）

- **成立时间：** 关于页**未写成立年份**，只写「自 2021 年起专注自研实时物理引擎」。外部报道更早：36氪 2021-08-16 写「成立于 2020 年 10 月」，CB Insights 亦记 2020 年。路线取 2020 年。
- **地点：** 北京总部（海淀区中关村创业大街）、杭州研发中心、武汉训练中心。
- **品牌口号 / 定位：** 「超越物理，进化不止」；仿真 → 数据 → 训练 → 真机部署闭环。
- **产品线：** MotrixSim（仿真引擎）、MotrixLab（模仿与强化学习训练框架）、MotrixGen（仿真数据合成，首页卡片指向联系页）、MotrixStudio（场景编辑器，含资产库与 AI Agent 助手）、SimReady 资产与资产工具链（MJCF 转换、几何调优、资产校验为免费工具；3DGS / 点云 / CAD / USD 转换与 AutoMJCF 为增值服务）。
- **迁移验证（首页自述）：** 训练策略已在 Unitree GO1 / G1、ANYmal-C 真机上迁移；算力适配 NVIDIA、AMD、Apple、鲲鹏、昇腾等。属应用案例描述，不代表正式合作。

## 成立时间外部佐证

[36氪](https://www.36kr.com/p/1355856359654023)（2021-08-16）原文写「成立于 2020 年 10 月」，[CB Insights](https://www.cbinsights.com/company/motphys) 亦记 2020 年；路线取 2020 年。本库只收技术节点，不收融资与商务合作信息。

## 版本日志（官网 release-notes，2026-10-17 共 9 个版本）

页面只列每版部分亮点，只写年月。**最早列出的 v0.2.0 不证明 MotrixSim 首发。**

| 版本 | 日期 | 亮点 |
| --- | --- | --- |
| v0.2.0 | 2025-09 | 多相机 RGBD 传感器、吸附驱动器、高度场加载、两个 Sim2Sim 示例 |
| v0.3.0 | 2025-09 | IK 模块、多环境并行、G1 复杂地形行走示例 |
| v0.4.0 | 2025-11 | DLS IK 求解器、Mocap 接口 |
| v0.5.0 | 2026-01 | G1 / Go2 键盘控制示例、运行时程序化组合模型、接触传感器与无头渲染 |
| v0.6.0 | 2026-03 | 正式支持 macOS、质量与质心随机化、OSC 控制器 |
| v0.7.0 | 2026-04 | 浏览器端 Web 平台上线、域随机化扩展 |
| v0.8.0 | 2026-05 | 全局光照渲染（SSGI）、运行时批量随机化 |
| v0.9.0 | 2026-07 | 3DGS 混合渲染、CoACD 凸分解、aarch64 Linux |
| v0.10.0 | 2026-08 | 多束 LiDAR、编辑器深色界面、HDR 环境光 |

## 研究页（2 条，均为多机构联合署名，页面未列个人作者）

| 日期 | 标题 | 入口 | 本库节点 |
| --- | --- | --- | --- |
| 2026.05 | UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms（清华 × Motphys 等 21 家机构） | arXiv:2605.30313；代码 `unilabsim/UniLab` | [UniLab](../../wiki/entities/unilab.md) |
| 2026.04 | GS-Playground: A High-Throughput Photorealistic Simulator for Vision-Informed Robot Learning（十余家机构，RSS 2026） | arXiv:2604.25459；代码 `discoverse-dev/gs_playground` | [GS-Playground](../../wiki/entities/gs-playground.md) |

页面写 UniLab 用 MotrixSim 作后端、端到端提速 3–10 倍；GS-Playground 渲染约 10,000 FPS、零样本视觉抓取成功率约 90%。均为页面转述的论文自报，本次未重读论文全文。

## 开放范围核查

- 官网只给出 GitHub 组织链接 <https://github.com/Motphys>，**未说明哪些项目开源、采用何种许可证**。
- 本库已归档 [MotrixLab](https://github.com/Motphys/MotrixLab) 仓库与 MotrixSim 文档（见 [motphys-motrix](../repos/motphys-motrix.md)）；MotrixSim 核心引擎是否开放源码本次**未确认**。
- MJCF「原生兼容 MuJoCo」只说明格式兼容，不等于开源。
- MotrixGen、MotrixStudio、增值资产服务需联系商务，无公开下载入口。

## 对 wiki 的映射

- 公司入口：[Motphys](../../wiki/entities/motphys.md)
- 路线节点：[Motrix](../../wiki/entities/motrix.md)、[GS-Playground](../../wiki/entities/gs-playground.md)、[UniLab](../../wiki/entities/unilab.md)
- 展示数据：`docs/company-roadmaps.json`；首页入口：`docs/index.html`

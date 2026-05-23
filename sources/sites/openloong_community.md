# OpenLoong 社区官网

> 来源归档

- **标题：** OpenLoong 社区 — 青龙·公版机项目页
- **类型：** site
- **来源：** 人形机器人（上海）有限公司 if applicable / OpenLoong 社区
- **链接：** https://www.openloong.org.cn/cn/projects/openloong
- **镜像入口：** https://www.openloong.org.cn/en （英文）
- **入库日期：** 2026-05-23
- **一句话说明：** 业内首个全尺寸开源人形「青龙」公版机的社区门户，聚合硬件 v2.5 设计资料、OpenLoong 控制框架、全身动力学控制包、训推数据集（白虎）及文档/活动/论坛入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/openloong.md`](../../wiki/entities/openloong.md)

---

## 为什么值得保留

- **全栈入口**：社区页将硬件、控制框架、动力学软件包、数据集与社区支持放在同一叙事下，是理解 OpenLoong 分层架构的读者友好入口。
- **公版机定位**：强调「强对标人」的全尺寸公版硬件 + 可裁剪的软件栈，与商业闭源整机形成对照。
- **配套资源索引**：文档中心、活动中心、社区论坛链接便于跟踪版本迭代与二次开发案例。

## 门户页摘录（2026-05 抓取）

| 板块 | 社区页描述 |
|------|------------|
| **硬件设计资料** | 本体设计方案与生产资料：结构、零部件规格、制造工艺；支持多版本迭代（页面标注 **v2.5**） |
| **OpenLoong 控制框架** | 操作系统核心组件：运动控制、感知处理、任务调度等完整控制程序库 |
| **全身动力学控制软件包** | 步态规划、平衡控制；面向复杂场景稳定行走 |
| **训推数据集** | 真机动作数据；两类典型末端、五大真实场景、30+ 高频任务类型 |
| **白虎数据集** | 与训推数据并列展示的社区数据集品牌 |
| **社区支持** | [文档中心](https://www.openloong.org.cn/cn/docs)、活动中心、社区论坛 |

## 官方延伸链接

- 项目总览 README：[loongOpen/OpenLoong](https://github.com/loongOpen/OpenLoong)
- 开放原子基金会项目页：[OpenLoong @ OpenAtom](https://www.openatom.org/project/ho1LksJyAPBU)
- 动力学控制 API / Wiki（官网托管）：[API 文档](https://www.openloong.org.cn/pages/api/html/index.html)、[Wiki](https://www.openloong.org.cn/pages/wiki/html/index.html)
- 联系邮箱：web@openloong.org.cn

## 与本仓库现有资料的关系

- 与 [`wiki/entities/open-source-humanoid-hardware.md`](../../wiki/entities/open-source-humanoid-hardware.md) 同属「可复现全尺寸人形」谱系；OpenLoong 强调 **国家级创新中心 + 开放原子** 运营与 **MPC/WBC + 自研 C++ 框架** 双轨软件。
- 与 [天工](./../../wiki/entities/tienkung-humanoid-open-source.md)、[Asimov v1](../../wiki/entities/asimov-v1.md) 并列：均为国内/国际开源人形参考，但 OpenLoong 是目前公开资料中 **唯一强调全尺寸 + 全栈图纸/控制/数据** 的公版方案之一。

## 对 wiki 的映射

- 升格 [`wiki/entities/openloong.md`](../../wiki/entities/openloong.md)：四层架构（云端大脑 / 具身小脑 / 具身实体 / 具身数据）、仓库矩阵与硬件子系统索引。
- 交叉更新 [`wiki/entities/open-source-humanoid-hardware.md`](../../wiki/entities/open-source-humanoid-hardware.md)、[`wiki/entities/humanoid-robot.md`](../../wiki/entities/humanoid-robot.md)。

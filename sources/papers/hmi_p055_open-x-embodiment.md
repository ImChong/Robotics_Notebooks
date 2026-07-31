# Open X-Embodiment: Robotic Learning Datasets and RT-X Models（Open X-Embodiment，HMI P055）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- **短名：** Open X-Embodiment
- **类型：** paper / hmi-papers / 世界模型、VLA与Agent
- **HMI ID：** P055
- **年份：** 2023
- **原文：** https://arxiv.org/abs/2310.08864
- **代码：** https://github.com/google-deepmind/open_x_embodiment
- **项目页：** https://robotics-transformer-x.github.io/
- **入库日期：** 2026-07-31
- **一句话说明：** 把 60+ 数据集、22 类本体整理到统一 schema，并用 RT-X 检验跨本体混合训练何时带来正迁移——统一的是存储与粗动作接口，不是动力学。
- **策展入口：** [HMI 论文与项目](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) · [逐篇解读 P055](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E8%AE%BA%E6%96%87%E9%80%90%E7%AF%87%E8%A7%A3%E8%AF%BB/P055.md)

## 开源状态（步骤 2.5）

- **结论：** 部分开源（数据协议与汇总入口；各源数据集许可仍独立）

## 摘录（编译自 HMI 解读，非原文复制）

### 摘录 1

Open X-Embodiment的价值先在数据基础设施，然后才在RT-X模型。60个数据集、22类本体、超过100万条轨迹被整理到相对统一的格式，使“跨机构、跨机器人联合训练”第一次有了足够大的公开实验底座。但这种统一主要发生在存储schema和常见末端动作层，并没有消除本体差异。

**对 wiki 的映射：** [`wiki/entities/paper-open-x-embodiment.md`](../../wiki/entities/paper-open-x-embodiment.md)

### 摘录 2

数据会被转为标准episode/步格式，包括图像、自然语言任务、机器人状态和动作。联合模型主要对齐为7维末端动作：位移、旋转和夹爪，再做数据集级归一化。但不同数据集的坐标系、绝对/相对动作、控制频率、相机位置、夹爪语义和任务分布仍然不同。这是“粗粒度共享动作空间”，不是把关节级动力学重定向问题解决了。

**对 wiki 的映射：** [`wiki/entities/paper-open-x-embodiment.md`](../../wiki/entities/paper-open-x-embodiment.md)

### 摘录 3

每个episode还应保留本体、数据源、任务文本、观测键、动作语义和时间结构，否则统一张量会掩盖不可比数据。RT-X训练通过数据集混合采样，让图像和语言条件映射到共享末端token；它没有一个显式跨本体世界模型去预测每台机器人的动力学。动作最终仍由各平台自己的逆运动学、轨迹控制与安全接口执行。

**对 wiki 的映射：** [`wiki/entities/paper-open-x-embodiment.md`](../../wiki/entities/paper-open-x-embodiment.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-open-x-embodiment.md`](../../wiki/entities/paper-open-x-embodiment.md)
- 覆盖索引：[`wiki/queries/hmi-papers-coverage.md`](../../wiki/queries/hmi-papers-coverage.md)
- 上游策展仓：[`sources/repos/humanoid-motion-intelligence.md`](../repos/humanoid-motion-intelligence.md)

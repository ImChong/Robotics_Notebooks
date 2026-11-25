# FastRoverVLA: Fast Vision-Language-Action Execution for Off-Road Autonomous Driving

- **作者：** Changtao Zhao, Yingxue Guan, Li Wang, Shijie Zhao, Yuhang Zhang, Shengkai Zhang, Qing Wang, Ziying Song
- **机构：** 北京理工大学、长安大学、北京交通大学、燕山大学
- **通讯作者：** Li Wang；项目负责人：Ziying Song（论文首页）
- **论文：** https://arxiv.org/abs/2611.26242
- **原文：** https://arxiv.org/html/2611.26242v1
- **PDF：** https://arxiv.org/pdf/2611.26242v1
- **版本：** arXiv:2611.26242v1，2026-11-19
- **项目：** https://adept-org.github.io/FastRoverVLA/
- **代码：** https://github.com/ADEPT-ORG/FastRoverVLA
- **入库日期：** 2026-11-24
- **核对范围：** arXiv v1 HTML 摘要、§1–5、方法公式、表1–8及附录 A 作者简介；核对项目页、README、execution.py、refresh.py、evaluation.py、reproduce.py；未执行作者的 CPU 或 GPU 实验。
- **通读：** 已完成

## 核心摘录（归纳）

- §3.1–3.2：冻结 VLM 输出速度–曲率序列，新计划与实测状态融合；每次更新再对齐、限制速度步长、平面积分并保留修正后尾部。
- §3.3：对未补齐动作的相邻差分进行归一化，使用短/长复用预算。预算不感知复用期间的环境变化。
- §4.1–4.2：636 个 ORAD-3D 与 650 个 RELLIS-3D 窗口，6 点、0.5 s 间隔。冻结预测重放与独立模型请求是不同协议。
- §4.3–4.4：移除状态对齐或写回显著增大误差；持续运动-only 控制在 RELLIS 上略优。等预算均匀刷新接近自适应；past-only heading 复评分保留收益。
- §4.5、§5：Qwen3-VL 独立串行请求处理成本每更新 355.4→21.4 ms、443.9→21.3 ms；不代表闭环延迟或车辆控制频率，实际闭环仍待验证。

## 对 wiki 的映射

- [FastRoverVLA 论文实体](../../wiki/entities/paper-fastrovervla.md)
- [项目页归档](../sites/fastrovervla.md)
- [源码归档](../repos/fastrovervla.md)

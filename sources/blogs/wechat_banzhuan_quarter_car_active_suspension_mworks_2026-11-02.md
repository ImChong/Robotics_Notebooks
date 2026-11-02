# 从 1/4 车模型到最优控制器：用 MWORKS 做一次主动悬架建模与参数优化

> 来源归档（blog / 微信公众号）

- **标题：** 从 1/4 车模型到最优控制器：用 MWORKS 做一次主动悬架建模与参数优化
- **类型：** blog
- **作者：** 搬砖工程师 / 算法工程师的学习日志（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/QXGYYRNvsxs7hfG2zwCK3A
- **发表日期：** 2026-11-02
- **入库日期：** 2026-11-02
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 13381 字符；公式在纯文本中丢失，页面里的公式由文字描述与 2.2–2.4 节的变量定义复原，未逐式核对
- **一句话说明：** 在 MWORKS.Sysplorer 用 Modelica 搭 1/4 车主动悬架，在 MWORKS.Syslab 设计天棚阻尼与全状态反馈、用 fmincon 优化增益，并跨工具校核；核心教训是「优化器只最优你写下的目标与约束」。
- **说明：** 这是单作者的建模 / 优化实操笔记，**没有论文或项目清单**，故不做逐项目节点；知识沉淀到 [quarter-car-active-suspension-lqr-optimization](../../wiki/concepts/quarter-car-active-suspension-lqr-optimization.md)。文中数值为作者自称在 MWORKS 2026b 上跑出（算例参数取自「公开文献常规取值」），本库未复现。主题是车辆主动悬架，与机器人的关联在于「状态反馈 / LQR / 带约束参数优化」的方法论。

## 文章结构

1. 主动悬架难点：三指标（车身加速度 / 悬架动行程 / 轮胎动载荷）互相冲突，作动器力与行程有限
2. 1/4 车辆模型：运动方程、四状态空间形式、RMS 指标、参数与两个模态频率（1.30 Hz / 11.80 Hz）
3. ISO 8608 路面谱：C 级路面、40 谐波离散、固定相位
4. Sysplorer / Modelica 建模：QuarterCar 组件复用、四控制器并行、求解器设置
5. Syslab：被动基线、谐波叠加稳态 RMS 加速优化
6. 天棚阻尼；LQR 手工权重的坑；fmincon 优化的「最优解陷阱」与补全约束
7. 跨工具校核（统计窗口长度）；结果分析（时域、频域、增益平面）；五个踩坑
8. 收尾结论

## 对 wiki 的映射

- 新页：[quarter-car-active-suspension-lqr-optimization](../../wiki/concepts/quarter-car-active-suspension-lqr-optimization.md)
- 回链：[LQR（形式化）](../../wiki/formalizations/lqr.md)

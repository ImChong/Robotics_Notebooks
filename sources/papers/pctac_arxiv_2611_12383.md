# PCTac（arXiv:2611.12383）

> 来源归档（paper）

- **标题：** PCTac: Mechanochromic Photonic-Crystal Tactile Sensor for Dexterous Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.12383>
- **PDF：** <https://arxiv.org/pdf/2611.12383v1>
- **项目页：** <https://bey0ndt1me.github.io/PCTac-page/>
- **代码：** 论文未给出代码仓库（摘要称 open-source，正文引言残留占位 `https://XXXX.github.io/`）；项目页仅列 CSV / JSON 数据资源，未见仓库链接
- **作者与机构：** Ruilin Zhang、Xuanshuo Liu 等；上海科技大学信息科学与技术学院、北京通用人工智能研究院（BIGAI）
- **入库日期：** 2026-11-11
- **一句话说明：** 用光子晶体受压变色的指尖作触觉传感器，把相对无载参考帧的色相变化场直接喂给 Diffusion Policy，在 SO-101 上三项真机任务比较视觉与触觉输入。
- **核对范围：** 通读 arXiv v1 PDF 全文 9 页（摘要、§I–§V、表 I–II、图 1–7 的图注与正文，参考文献仅浏览）；项目页静态读取页面文字与结果表。论文无附录 / 补充材料；图像内容与项目页视频未逐帧查看；未找到代码仓库，故未核对代码。
- **通读：** 已完成

## 核心摘录

1. **传感原理（§III-A）：** 光子晶体按 Bragg–Snell 关系选择性反射，λpeak = 2d·sqrt(neff² − sin²θ)，法向入射近似 2·neff·d；受压改变周期结构即改变反射色。用 RGB 相机记录，不测光谱。
2. **结构（§III-B）：** 仿 GelSight Wedge 外壳（白色 PLA），OV5647 相机加反光镜折叠光路，白光 LED；表层黑色硅胶墨水，下为 20 mm × 30 mm、厚 15 µm 的 PC 膜，再下为 3 mm 厚 Solaris 弹性体（Shore A 15）。
3. **表征（§IV-A，图 4）：** 5 mm 球形压头，22 次法向加载试验，色相随载荷明显变化，加载 / 卸载有滞后；该压头下约 1.5 N 饱和，更大接触面可测更高载荷。
4. **8 种物体（§IV-B，图 5）：** 10 N 固定法向载荷；凸起与细纹可辨，花生等凹陷特征响应弱。
5. **模仿学习（§IV-C，表 II）：** 每任务 60 条遥操作演示，每策略每任务 30 次真机 rollout；总成功率 PCTac Hue 68/90（75.6%）、PCTac RGB 60/90（66.7%）、Wedge RGB 57/90（63.3%）、纯视觉 29/90（32.2%）。
6. **局限（§IV-D）：** 凹面灵敏度低；PC 主要响应局部应力，法向与剪切组合可产生相似色图。

## 对 wiki 的映射

- [paper-pctac](../../wiki/entities/paper-pctac.md)

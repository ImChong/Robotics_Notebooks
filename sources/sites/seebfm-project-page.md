# SeeBFM 官方项目页与公开视频说明

> 来源归档（ingest；项目首页主体为超大生成 HTML，已逐页核对可读的公开视频配音/幻灯片源码，论文与首页正文仍待补查）

- **标题：** SeeBFM: Behavior Foundation Model That Sees
- **项目页：** <https://capfab.github.io/seebfm-media/>
- **交互式演示：** <https://capfab.github.io/seebfm-media/playground/>
- **作者主页：** <https://anindex.github.io/>
- **网站素材仓库：** <https://github.com/capfab/seebfm-media>
- **论文：** 作者主页仅标 “Under review · 2026”；未找到可直接通读的论文全文链接。
- **代码：** 公开的是网站、视频与 3D 仿真回放浏览器源码，不是 SeeBFM 的策略训练、推理、checkpoint 或真机部署代码。
- **数据：** 网站 `playground/data/` 含供演示读取的已录制仿真轨迹、机器人几何与地形网格；未证实有可用于重训的原始训练数据集。
- **核查日期：** 2026-11-11
- **核对范围：** 项目目录结构、README、完整 `slides/spec.js`、`slides/slots.js`、`media/overview.vtt`、`playground/index.html` 与 `playground/viewer.js`、`playground/match.js`。项目首页 `index.html` 是 2.2 MB 的生成产物，本次读取工具没有返回正文；论文全文亦未取得。
- **通读：** 未完成

## 项目页验证要点

- **行为接口：** 同一策略接受由 reward / goal pose / motion 产生的 prompt；另有 typed language request 经语言匹配转为已选的运动 prompt。
- **训练结构：** BFM-Zero 的地形高度扫描后训练；backward network 不读取地形，其他网络读取，以守住意图表达不变、执行随地形调整的边界。
- **相机方案：** 两台 depth camera 的学生网络回归高度扫描，替代控制器要求的 scan input，原控制器冻结。
- **演示：** 上下楼梯、箱体、缺口、移动障碍、受推恢复、目标姿态以及文字指令；所有结果均在仿真。
- **数据口径：** 项目讲解分镜 `slides/spec.js` 报告了 54.4%/41.0%/68% 等穿越率。仅代表作者公开展示数据，尚未核实完整论文统计。
- **局限：** 官方讲解明确 45 cm 障碍会停在前方；Unitree G1 搭载相机学生的真实机器人测试尚未完成。
- **回放不是在线仿真：** Playground 用预录的 Isaac Lab 运动轨迹和 Three.js 渲染；不能在页面交互中动态改变机器人策略。

## 交叉引用

- [论文来源核对与待办](../papers/seebfm_preprint_2026.md)
- [展示源码核对](../repos/seebfm-media-page-source.md)
- [SeeBFM 知识节点](../../wiki/entities/paper-seebfm.md)

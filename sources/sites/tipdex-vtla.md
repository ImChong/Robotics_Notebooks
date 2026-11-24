# TipDex-VTLA 项目页

> 官方项目来源归档

- **类型：** site
- **链接：** https://tiplab.github.io/TipDex-VTLA/
- **论文：** https://arxiv.org/abs/2611.25683v2
- **代码：** https://github.com/tiplab/TipDex-VTLA
- **权重：** https://huggingface.co/TipDex-VTLA/TipDex-VTLA
- **数据集：** https://huggingface.co/datasets/TipDex-VTLA/TipDex-Data
- **入库日期：** 2026-11-24
- **开源状态：** 部分开源，推理代码、鸡蛋转移模型及配对数据已公开；不等同四任务训练全集。

## 核查要点

- 项目页列出四种光照与触觉联合修正情形，2 Hz 慢分支与 30 Hz 快分支。
- 真机展示鸡蛋转移、葡萄捏取、纸巾抽取、绿豆倾倒；仿真为 UniVTAC。
- HF 模型卡列出 Orbax 基础策略、VASE、TACE、共享编码器预测器及样例。
- HF 数据卡为鸡蛋入盒任务，50 个原始轨迹和 50 个 RGB 合成光照变体，94,172 帧、30 fps；变体不是新增物理示教。

## 对 wiki 的映射

- [TipDex-VTLA](../../wiki/entities/paper-tipdex-vtla.md)
- [论文归档](../papers/tipdex_vtla_arxiv_2611_25683.md)
- [源码归档](../repos/tipdex-vtla.md)

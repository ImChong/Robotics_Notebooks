# LIBERO-Recovery（liulin815/LIBERO-Recovery）

- **URL：** <https://github.com/liulin815/LIBERO-Recovery>
- **组织：** liulin815
- **关联论文：** [libero_recover_arxiv_2609_05178](../papers/libero_recover_arxiv_2609_05178.md)
- **项目页：** <https://liulin815.github.io/LIBERO-Recovery/>
- **数据：** ModelScope `ataier/LIBERO_Recovery_Assets`（评测必需）、`LIBERO_Recovery_Expert`（训练）、`LIBERO_10_LL`（失败前历史）
- **实体页：** [paper-libero-recover](../../wiki/entities/paper-libero-recover.md)

## 一句话说明

LIBERO-Recover 评测管线：launcher `eval_libero_custom_scene.sh` 启 policy server + MuJoCo client；每场景 10 rollouts（5 夹爪闭合 + 5 默认开），±2 cm xy 扰动；需下载 `LIBERO_Recovery_Assets` 到 `$STARVLA/assets/scenes/`。

## 交叉链接

- [LIBERO-Recover 论文实体](../../wiki/entities/paper-libero-recover.md)
- [LIBERO-Recover 项目页](../sites/libero-recovery-github-io.md)

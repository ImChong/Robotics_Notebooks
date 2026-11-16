# aus.bot/research/autointervene — AutoIntervene 项目官网与 GitHub Pages 仓库核查

> 项目官方网站归档

- **官方 URL：** <https://aus.bot/research/autointervene/>
- **镜像静态仓库（未在论文外链，仅作者静态站点源）：** `github.com/123qwedsa123/AutoIntervene`（commit `e71f1c8`，提交者 `jt2079` 即第一作者 Jinhe Tang）
- **对应论文：** [arXiv:2608.07065 — AutoIntervene](../papers/autointervene_arxiv_2608_07065.md)
- **通读：** 已完成
- **核对范围：** 已通读 `originals/sites/autointervene.html`、`autointervene_alt.html`（内容逐字一致）以及 `originals/repos/123qwedsa123__AutoIntervene/repo` 全仓 28 个文件（`README.md`、`index.html`、`styles.css`、`assets/*`），核实网站表格数据与代码发布状态。

---

## 站点核对与开源真相查明

1. **官网内容一致性：** `aus.bot/research/autointervene/` 展示方法四步流程（Construct / Evaluate / Transfer / Adapt）、9 项双臂真机实验 MP4 演示视频与完整 Tables 1–4 定量表格，全部数值与论文 `arXiv:2608.07065` 对应表格完全一致（官网 Table 1–4 依次对应论文 Table II–V）。
2. **关于 GitHub 仓库 `123qwedsa123/AutoIntervene` 的核实：**
   - 经比对 `123qwedsa123/AutoIntervene/index.html` 与官网 `autointervene.html`，两者仅差官网 `<link rel="canonical" href="https://aus.bot/research/autointervene/">` 与 Cloudflare 统计脚本两行，`README.md` 明确写明 `"Static academic project page for AutoIntervene"`。
   - 该仓库仅托管静态网页 HTML/CSS、论文 PDF (`assets/AutoIntervene.pdf`) 与 9 段压缩 MP4 演示视频，**不包含任何 Python 源码、DINOv3 记忆检索监控器实现、ACT/DP/FM 策略代码或硬件遥操脚本**，且论文正文中并未将其作为代码库外链。
3. **最终开源结论：** **算法与策略代码未开源**，请勿将 `123qwedsa123/AutoIntervene` 静态站点仓库误认为可运行的算法实现。

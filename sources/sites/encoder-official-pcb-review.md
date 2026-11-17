# RLS、HEIDENHAIN 与 TI 编码器官方资料复核

- **类型：** 官方技术资料 / 原文复核
- **核对日期：** 2026-11-16
- **核对范围：** RLS AksIM-2 产品页完整技术正文与参数/下载目录；HEIDENHAIN Rotary encoders 分类页全文；TI LDC5072-Q1 的完整 Product details、Features、Description 技术正文。未声称读完下载目录的手册、封闭安全文档或完成编码器标定。
- **通读：** 已完成

## 原文与版本

- [RLS AksIM-2](https://www.rls.si/eng/aksim-2-off-axis-rotary-absolute-encoder)；本次读取材料 SHA-256：`f97b923125667b8b2fa547a018bf57c4aee949c7fde14a16d6493ec75d449d5a`。
- [HEIDENHAIN rotary encoders](https://www.heidenhain.com/products/encoders/rotary-encoders)；本次读取材料 SHA-256：`80113bfd27611080569c5a3c45d3714a55372bee429d37e329b8545f239d2f27`。
- [TI LDC5072-Q1](https://www.ti.com/product/LDC5072-Q1)；本次读取材料 SHA-256：`059f0077fd9988933a7b1fcdd8fcf59042e5a6947bf01d408e41aa6b132bb4f1`。

## 核对结论

- AksIM-2 使用轴向磁化磁环和读头；页面分别给出最高 20 bit 分辨率、校准后 ±0.004°–±0.020° 系统精度，不能把位数直接当准确度。
- HEIDENHAIN 分类页区分外装与内置编码器的防护及工作温度，这些是不同安装类别的指标，不能把分类最大值混用到同一个器件。
- LDC5072-Q1 激励 PCB 线圈，导电靶改变耦合到两组接收线圈的信号，输出位置的正弦/余弦分量；不需要永磁体。芯片前端的能力不是任意线圈和安装后的整机精度。
- 旧微信文章仍保留为线索，不补写“通读已完成”；本次按厂商明确的器件与系统边界修订横向参数表。

## 对 wiki 的映射

- [joint-encoder-selection](../../wiki/concepts/joint-encoder-selection.md)

# SymSkill 官方代码

- **Repo：** <https://github.com/shaoyifei96/Symskill>
- **对应论文：** <https://arxiv.org/abs/2510.01661>
- **项目页：** <https://symskill.github.io/>
- **开源状态：** 已开源（需四仓工作区和 MOSEK license）
- **入口核对：** `predicators/main.py`（环境与 approach）→ `predicators/approaches/grammar_search_invention_approach.py`（谓词搜索）→ `predicators/nsrt_learning/nsrt_learning_main.py`（算子学习）→ `predicators/planning.py`（规划执行）；测试 `pytest predicators_robocasa/tests/test_sym_skill.py`。
- **论文详情：** [SymSkill](../../wiki/entities/paper-symskill.md)

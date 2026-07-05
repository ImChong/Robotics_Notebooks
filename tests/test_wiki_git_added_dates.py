"""wiki_git_added_dates：git 首次入库日期（action 标签兜底数据源）。"""

from __future__ import annotations

import unittest
from unittest import mock

import generate_link_graph as glg


class WikiGitAddedDatesTest(unittest.TestCase):
    def setUp(self) -> None:
        glg._WIKI_GIT_ADDED_DATES_CACHE = None
        self.addCleanup(lambda: setattr(glg, "_WIKI_GIT_ADDED_DATES_CACHE", None))

    def test_collects_first_add_date(self) -> None:
        log_text = (
            f"{glg._GIT_LOG_BOUNDARY}2026-05-28\n"
            "M\twiki/concepts/sim2real.md\n"
            f"{glg._GIT_LOG_BOUNDARY}2026-05-01\n"
            "A\twiki/concepts/sim2real.md\n"
        )
        paths = ["wiki/concepts/sim2real.md"]

        def fake_run(*_args, **_kwargs):
            return mock.Mock(returncode=0, stdout=log_text)

        with (
            mock.patch.object(glg, "_iter_wiki_md_paths", return_value=paths),
            mock.patch("generate_link_graph.subprocess.run", side_effect=fake_run),
        ):
            out = glg.wiki_git_added_dates(force_refresh=True)

        self.assertEqual(out["wiki/concepts/sim2real.md"], "2026-05-01")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

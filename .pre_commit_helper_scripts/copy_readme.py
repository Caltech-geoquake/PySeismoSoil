"""Copy ``README.md`` into the Sphinx documentation entry page."""  # noqa: INP001

from __future__ import annotations

from pathlib import Path

_README = Path('README.md')
_DOCS_INDEX = Path('docs/source/index.md')

# Appended after the README so the class and module pages stay in the
# documentation sidebar. The directive is hidden, so the rendered homepage
# is the README. It is not part of README.md: GitHub would show the fence.
_API_NAV = '\n```{toctree}\n:hidden:\n:maxdepth: 2\n\nclasses\nmodules\n```\n'


def main() -> None:
    """Write the README, plus the API toctree, to ``docs/source/index.md``."""
    text = _README.read_text(encoding='utf-8')
    if not text.endswith('\n'):
        text += '\n'

    _DOCS_INDEX.write_text(text + _API_NAV, encoding='utf-8')


if __name__ == '__main__':
    main()

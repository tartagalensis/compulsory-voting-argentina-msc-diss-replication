"""
format_tables.py
----------------
Post-process the LaTeX tables exported to outputs/tables/ so they render
centered and fit within the text margins in the dissertation.

The notebooks export tables via pandas ``to_latex`` wrapped in a ``table``
environment, but without ``\\centering`` or any width control, so wide tables
overflow the right margin and all of them sit left-aligned. This script adds, to
each ``*.tex`` table:

  * ``[ht]`` float placement and ``\\centering``;
  * a ``\\resizebox`` that shrinks the tabular to ``\\linewidth`` only when it is
    wider than the text block (narrow tables keep their natural size).

The transformation is idempotent: files that already contain ``\\centering`` are
left untouched, so it is safe to re-run after regenerating any table.

Run:
    python src/format_tables.py
"""

from pathlib import Path

TABLES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"

# Shrink to the text width only if the natural width exceeds it.
RESIZE_OPEN = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"


def format_table_file(path: Path) -> bool:
    """Center and width-fit a single table file. Returns True if modified."""
    s = path.read_text()
    if r"\centering" in s:
        return False  # already processed
    if not s.lstrip().startswith(r"\begin{table}"):
        return False  # not a table float
    s = s.replace("\\begin{table}\n", "\\begin{table}[ht]\n\\centering\n", 1)
    s = s.replace("\\begin{tabular}", RESIZE_OPEN + "\n\\begin{tabular}", 1)
    s = s.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.write_text(s)
    return True


def main():
    changed = [p.name for p in sorted(TABLES_DIR.glob("*.tex")) if format_table_file(p)]
    if changed:
        print(f"Formatted {len(changed)} table(s):")
        for name in changed:
            print(f"  {name}")
    else:
        print("All tables already formatted (nothing to do).")


if __name__ == "__main__":
    main()

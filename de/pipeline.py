"""DuckDB + dbt structured-source pipeline: build the warehouse with dbt, then
export the `documents` mart as a text corpus for the RAG ingestion."""

import shutil
import subprocess
from pathlib import Path

DBT_DIR = Path("de/dbt")
DUCKDB_PATH = DBT_DIR / "warehouse.duckdb"


def run_dbt(project_dir=DBT_DIR) -> None:
    """Run `dbt build` in the dbt project. Raises if dbt isn't installed or fails."""
    if shutil.which("dbt") is None:
        raise RuntimeError(
            "dbt CLI not found — install the de extra: pip install -e '.[de]'"
        )
    result = subprocess.run(
        ["dbt", "build", "--project-dir", ".", "--profiles-dir", "."],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dbt build failed:\n{result.stdout}\n{result.stderr}")


def export_documents(duckdb_path=DUCKDB_PATH) -> list[dict]:
    """Read the dbt `documents` mart from DuckDB."""
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "duckdb not installed — install the de extra: pip install -e '.[de]'"
        ) from exc
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT id, source, content FROM documents ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    return [{"id": r[0], "source": r[1], "content": r[2]} for r in rows]


def write_corpus(rows, out_dir="data/warehouse_corpus") -> int:
    """Write each document row to a .md file the RAG ingestion can consume."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        slug = str(row["source"]).replace(":", "_").replace("/", "_")
        (out / f"{slug}.md").write_text(row["content"], encoding="utf-8")
    return len(rows)

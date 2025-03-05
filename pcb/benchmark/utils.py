"""Utilities"""

from pathlib import Path


def jid_to_json_path(jid: str, output_dir: str | Path) -> Path:
    """
    Converts a job ID to a JSON file path that looks like

        output_dir / jobs / jid[:2] / jid[2:4] / jid.json

    """
    return (
        Path(output_dir)
        / "jobs"
        / jid[:2]  # spread files in subdirs
        / jid[2:4]
        / f"{jid}.json"
    )

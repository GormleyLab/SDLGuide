#!/usr/bin/env python3
"""
Fix metadata.widgets so GitHub/nbconvert can render the notebook while keeping
widget values (dropdowns, text, etc.).

GitHub expects: metadata.widgets[MIMETYPE]["state"] to exist.
We wrap the existing widget state in that structure if it's missing.
"""

import json
import sys

WIDGET_STATE_MIMETYPE = "application/vnd.jupyter.widget-state+json"


def _notebook_has_widget_references(nb):
    """True if any cell has widget views in outputs or referenced_widgets in metadata."""
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            data = out.get("data") or {}
            if "application/vnd.jupyter.widget-view+json" in data:
                return True
        refs = (cell.get("metadata") or {}).get("referenced_widgets")
        if refs:
            return True
    return False


def fix_notebook(notebook_path: str) -> bool:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    meta = nb.get("metadata") or {}
    widgets = meta.get("widgets") or {}

    # If notebook has widget references but no widget state metadata, add minimal valid structure
    if WIDGET_STATE_MIMETYPE not in widgets:
        if _notebook_has_widget_references(nb):
            meta["widgets"] = {
                WIDGET_STATE_MIMETYPE: {
                    "state": {
                        "version_major": 2,
                        "version_minor": 0,
                        "state": {},
                    }
                }
            }
            nb["metadata"] = meta
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Added metadata.widgets (with 'state' key) for GitHub. Updated: {notebook_path}")
            return True
        print("No widget state in notebook. Nothing to fix.")
        return False

    entry = widgets[WIDGET_STATE_MIMETYPE]

    # Already has top-level "state" key and it looks like the full document
    if "state" in entry:
        inner = entry["state"]
        # If it's the full doc (has version_major or a nested "state" with widget IDs), we're good
        if isinstance(inner, dict) and ("version_major" in inner or "state" in inner):
            print("Widget metadata already in correct format.")
            return False
        # If "state" is the raw widget map (keys look like IDs), wrap it
        if isinstance(inner, dict) and inner and _looks_like_widget_map(inner):
            wrapped = wrap_widget_state(inner)
            widgets[WIDGET_STATE_MIMETYPE] = {"state": wrapped}
        else:
            print("Widget metadata structure unclear. Leaving as-is.")
            return False
    else:
        # Widget IDs are directly under the MIME type (wrong format)
        wrapped = wrap_widget_state(entry)
        widgets[WIDGET_STATE_MIMETYPE] = {"state": wrapped}

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Updated widget metadata in: {notebook_path}")
    return True


def _looks_like_widget_map(obj):
    if not obj or not isinstance(obj, dict):
        return False
    keys = list(obj.keys())[:5]
    for k in keys:
        if isinstance(k, str):
            if len(k) == 32 and all(c in "0123456789abcdef" for c in k.lower()):
                return True
            if k.startswith("IPY_MODEL_"):
                return True
    return False


def wrap_widget_state(widget_id_to_model: dict) -> dict:
    """Wrap widget id -> model map in the full widget state document."""
    if "version_major" in widget_id_to_model and "state" in widget_id_to_model:
        return widget_id_to_model
    return {
        "version_major": 2,
        "version_minor": 0,
        "state": widget_id_to_model,
    }


if __name__ == "__main__":
    path = r"Notebook 2 - SDL Tutorial.ipynb"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    fix_notebook(path)

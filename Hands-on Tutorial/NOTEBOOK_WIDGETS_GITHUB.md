# Notebook widget metadata format for GitHub

GitHub (and nbconvert) require a specific structure for `metadata.widgets`. If the structure is wrong, you get:

> Invalid Notebook: the 'state' key is missing from 'metadata.widgets'. Add 'state' to each, or remove 'metadata.widgets'.

## Format GitHub expects

Each entry under `metadata.widgets` (e.g. `application/vnd.jupyter.widget-state+json`) **must** have a top-level **`state`** key. The full structure is:

```json
"metadata": {
  "widgets": {
    "application/vnd.jupyter.widget-state+json": {
      "state": {
        "version_major": 2,
        "version_minor": 0,
        "state": {
          "WIDGET_ID_1": {
            "model_module": "@jupyter-widgets/controls",
            "model_module_version": "1.5.0",
            "model_name": "TextModel",
            "state": { "value": "", "_dom_classes": [], ... }
          },
          "WIDGET_ID_2": { ... }
        }
      }
    }
  }
}
```

- **Outer `state`** (required): nbconvert looks for `metadata.widgets[MIMETYPE]["state"]`. This key must exist.
- **Inner object**: the standard ipywidgets state document with `version_major`, `version_minor`, and `state` (the map of widget id → model state). That keeps dropdown choices, text values, etc.

## Wrong format (causes the error)

Widget IDs directly under the MIME type, with no `state` wrapper:

```json
"widgets": {
  "application/vnd.jupyter.widget-state+json": {
    "001604a0b00849d8be084a796c01d308": { "model_name": "TextModel", "state": { ... } },
    "040786107b1a435798bc108b0d9accc2": { ... }
  }
}
```

## Fixing a notebook

From the repo root, run (PowerShell on Windows):

```powershell
python "Hands-on Tutorial/fix_widget_metadata_for_github.py" "Hands-on Tutorial/Notebook 2 - SDL Tutorial.ipynb"
```

Or from inside the `Hands-on Tutorial` folder:

```powershell
cd "Hands-on Tutorial"
python fix_widget_metadata_for_github.py "Notebook 2 - SDL Tutorial.ipynb"
```

That script rewraps existing widget metadata into the correct shape so GitHub can render the notebook and widget values are preserved.

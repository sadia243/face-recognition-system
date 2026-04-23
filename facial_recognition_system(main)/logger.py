"""
logger.py  —  CSV logging for the celebrity face recognition system.

Log file: results/recognition_log.csv

Columns
-------
Timestamp        : date and time of the recognition event
Image_Name       : filename of the uploaded photo (basename only)
Predicted_Person : top-1 predicted celebrity name
Confidence       : probability score (0.0000 – 1.0000)
Status           : "Identified" or "Low Confidence"

Public functions
----------------
log_result(image_path, results)  -> None
show_log()                       -> None  (prints log to console)
clear_log()                      -> None  (deletes the log file)
"""

import csv
import os
from datetime import datetime

os.makedirs('results', exist_ok=True)

# ── Constants ─────────────────────────────────────────────────
LOG_PATH = os.path.join('results', 'recognition_log.csv')
FIELDS   = ['Timestamp', 'Image_Name', 'Predicted_Person', 'Confidence', 'Status']


def log_result(image_path, results):
    """
    Append the top recognition result to the CSV log.

    Parameters
    ----------
    image_path : str
        Full or relative path to the uploaded photo.
    results : list[dict]
        Output from identify_person() — only results[0] (top match) is logged.
        Each dict must contain: name, confidence, low_conf.
    """
    if not results:
        return

    top = results[0]  # log only the best match per upload

    # Determine status label from the low_conf flag set by identify_person()
    status = 'Low Confidence' if top.get('low_conf') else 'Identified'

    is_new_file = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        # Write header row only the first time the file is created
        if is_new_file:
            writer.writeheader()

        writer.writerow({
            'Timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Image_Name':       os.path.basename(image_path),
            'Predicted_Person': top['name'],
            'Confidence':       f"{top['confidence']:.4f}",
            'Status':           status,
        })


def show_log():
    """
    Print the contents of the recognition log to the console.
    If the log does not exist, prints an informative message.
    """
    if not os.path.exists(LOG_PATH):
        print(f"No log file found at '{LOG_PATH}'.")
        print("Run the recognition system on some photos first.")
        return

    print(f"\n{'=' * 70}")
    print(f"Recognition Log  —  {LOG_PATH}")
    print(f"{'=' * 70}")

    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    if not rows:
        print("(log file is empty)")
        return

    # Column widths for pretty-printing
    col_widths = {field: max(len(field), max(len(r[field]) for r in rows))
                  for field in FIELDS}

    # Header
    header = '  '.join(f.ljust(col_widths[f]) for f in FIELDS)
    print(header)
    print('-' * len(header))

    for row in rows:
        print('  '.join(row[f].ljust(col_widths[f]) for f in FIELDS))

    print(f"\nTotal entries: {len(rows)}")
    print(f"{'=' * 70}\n")


def clear_log():
    """
    Delete the recognition log file.
    Prints a confirmation message or a warning if nothing to delete.
    """
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
        print(f"Log cleared: '{LOG_PATH}' has been deleted.")
    else:
        print(f"Nothing to clear — '{LOG_PATH}' does not exist.")


# ── Quick test when run directly ──────────────────────────────
if __name__ == '__main__':
    show_log()

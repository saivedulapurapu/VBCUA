"""
Backfills the delivery measures (speaking rate, long pauses, longest pause) for attempts
stored before they were recorded, from each attempt's saved audio file.

Usage (from the repo root):
    python research/backfill_delivery_measures.py            # update the database
    python research/backfill_delivery_measures.py --dry-run  # only print what would change

Only rows where speaking_rate_wpm is empty and the audio file still exists are changed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.database import SessionLocal  # noqa: E402
from backend.app.main import migrate_evaluations_table  # noqa: E402
from backend.app.models.evaluation import Evaluation  # noqa: E402
from backend.app.services.audio_analysis import measure_delivery  # noqa: E402

FIELDS = ("speaking_rate_wpm", "long_pause_count", "long_pauses_per_min", "longest_pause_s")


def backfill(session_factory, dry_run: bool = False) -> list:
    """Returns one (id, status, measures) tuple per attempt that needed measures."""
    migrate_evaluations_table(session_factory)  # make sure the columns exist
    db = session_factory()
    results = []
    try:
        for row in db.query(Evaluation).filter(Evaluation.speaking_rate_wpm.is_(None)).order_by(Evaluation.id):
            if not row.audio_path or not os.path.exists(row.audio_path):
                results.append((row.id, "audio missing", None))
                continue
            measures = measure_delivery(row.audio_path, row.transcription)
            if not dry_run:
                for field in FIELDS:
                    setattr(row, field, measures[field])
            results.append((row.id, "dry run" if dry_run else "updated", measures))
        if not dry_run:
            db.commit()
    finally:
        db.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the measures without saving them")
    args = parser.parse_args()
    results = backfill(SessionLocal, dry_run=args.dry_run)
    if not results:
        print("Nothing to backfill: every attempt already has delivery measures.")
    for eval_id, status, m in results:
        detail = "" if m is None else (
            f"{m['speaking_rate_wpm']} wpm, {m['long_pause_count']} long pauses "
            f"({m['long_pauses_per_min']}/min), longest {m['longest_pause_s']} s"
        )
        print(f"#{eval_id}: {status} {detail}")


if __name__ == "__main__":
    main()

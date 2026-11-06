#!/usr/bin/env python3
"""Turn downloaded snapshots into the data files behind the website.

Reads snapshots/ (never committed) and writes docs/data/ (committed). The output holds
only paper titles, paper numbers, forum ids, primary areas, keywords, and numeric scores.
Review text, abstracts, authors and reviewer ids are never copied,
and check_output() enforces this on the final files.

Each snapshot leaves a small numeric summary in docs/data/snapshots/, so the score
history survives even after the raw snapshot is deleted.
"""
import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common import iter_notes, reviews_of, value

FIELDS = ["rating", "confidence", "soundness", "presentation", "informativeness"]
TOP_SHARES = [0.05, 0.10, 0.25, 0.50]
EXCLUDED_STATUSES = {"withdrawn", "desk_rejected"}


def to_number(raw):
    """Read a score given as a number or as text such as "6: marginally above"."""
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return raw
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(raw))
    if not match:
        return None
    number = float(match.group(1))
    return int(number) if number.is_integer() else number


def status_of(note):
    venueid = value(note["content"].get("venueid")) or ""
    if venueid.endswith("/Withdrawn_Submission"):
        return "withdrawn"
    if venueid.endswith("/Desk_Rejected_Submission"):
        return "desk_rejected"
    return "active"


def sanitize_text(text):
    return " ".join(str(text or "").split())


def extract_keywords(raw):
    val = value(raw)
    if isinstance(val, list):
        return [sanitize_text(k) for k in val if sanitize_text(k)]
    if isinstance(val, str):
        return [sanitize_text(k) for k in val.split(",") if sanitize_text(k)]
    return []


def read_snapshot(snapshot_dir):
    """Extract only the allowed fields from a raw snapshot."""
    papers, statuses = [], Counter()
    for note in iter_notes(snapshot_dir):
        status = status_of(note)
        statuses[status] += 1
        if status in EXCLUDED_STATUSES:
            continue
        scores = {field: [] for field in FIELDS}
        for review in reviews_of(note):
            for field in FIELDS:
                number = to_number(value(review["content"].get(field)))
                if number is not None:
                    scores[field].append(number)
        primary_area = sanitize_text(value(note["content"].get("primary_area"))) or "Uncategorized"
        keywords = extract_keywords(note["content"].get("keywords"))
        papers.append({
            "number": note["number"],
            "forum": note["forum"],
            "title": sanitize_text(value(note["content"].get("title"))),
            "primary_area": primary_area,
            "keywords": keywords,
            "scores": scores,
        })
    papers.sort(key=lambda p: p["number"])
    return papers, statuses


def mean(numbers):
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def summarize(snapshot_id, taken_at, papers, statuses):
    """Build the numeric summary of one snapshot. It holds no text."""
    hists = {f: Counter(s for p in papers for s in p["scores"][f]) for f in FIELDS}
    means = sorted((mean(p["scores"]["rating"]) for p in papers if p["scores"]["rating"]),
                   reverse=True)
    all_ratings = [s for p in papers for s in p["scores"]["rating"]]
    mean_counts = Counter(means)
    cutoffs = {f"{share:.2f}": means[max(0, math.ceil(share * len(means)) - 1)]
               for share in TOP_SHARES} if means else {}
    return {
        "id": snapshot_id,
        "taken_at": taken_at,
        "submissions": sum(statuses.values()),
        "active": statuses["active"],
        "withdrawn": statuses["withdrawn"],
        "desk_rejected": statuses["desk_rejected"],
        "papers_with_ratings": len(means),
        "reviews": len(all_ratings),
        "mean_rating": mean(all_ratings),
        "mean_of_paper_means": mean(means),
        "median_paper_mean": round(statistics.median(means), 4) if means else None,
        "top_share_cutoffs": cutoffs,
        "histograms": {f: [[k, hists[f][k]] for k in sorted(hists[f])]
                       for f in FIELDS if hists[f]},
        "mean_distribution": [[m, mean_counts[m]] for m in sorted(mean_counts)],
        "ratings_by_paper": {str(p["number"]): sorted(p["scores"]["rating"], reverse=True)
                             for p in papers},
    }


def rows_for_page(papers, first_ratings):
    rows = []
    for p in papers:
        ratings = sorted(p["scores"]["rating"], reverse=True)
        rows.append([
            p["number"],
            p["forum"],
            p["title"],
            p["primary_area"],
            p["keywords"],
            ratings,
            mean(p["scores"]["confidence"]),
            mean(first_ratings.get(str(p["number"])) or []),
        ])
    return rows


def check_output(site, papers_doc):
    """Stop the build if any unexpected structure or forbidden text reached the page data."""
    allowed_site = {"venue", "generated_at", "latest", "fields", "primary_areas", "snapshots"}
    allowed_papers = {"columns", "rows"}
    if set(site) != allowed_site or set(papers_doc) != allowed_papers:
        raise ValueError(f"unexpected keys in page data: site={set(site)}, papers={set(papers_doc)}")
    expected_cols = ["number", "forum", "title", "primary_area", "keywords", "ratings", "confidence_mean", "first_mean"]
    if papers_doc["columns"] != expected_cols:
        raise ValueError(f"unexpected columns: {papers_doc['columns']}")
    if not isinstance(site.get("primary_areas"), list):
        raise ValueError("site.json must contain primary_areas list")
    for area in site["primary_areas"]:
        if not isinstance(area, str) or not area:
            raise ValueError(f"invalid primary area in site list: {area}")

    for row in papers_doc["rows"]:
        number, forum, title, primary_area, keywords, ratings, confidence, first_mean = row
        ok = (isinstance(number, int)
              and re.fullmatch(r"[\w-]+", forum)
              and isinstance(title, str)
              and isinstance(primary_area, str)
              and isinstance(keywords, list)
              and all(isinstance(k, str) for k in keywords)
              and all(isinstance(r, (int, float)) for r in ratings)
              and all(v is None or isinstance(v, (int, float))
                      for v in (confidence, first_mean)))
        if not ok:
            raise ValueError(f"unexpected row for paper {number}: {row}")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--snapshots", type=Path, default=Path("snapshots"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    data_dir = args.docs / "data"
    summary_dir = data_dir / "snapshots"
    summary_dir.mkdir(parents=True, exist_ok=True)

    usable = []
    for manifest_path in sorted(args.snapshots.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("complete") or args.allow_incomplete:
            usable.append((manifest_path.parent, manifest))
        else:
            print(f"Skipping incomplete snapshot {manifest_path.parent.name}", file=sys.stderr)
    if not usable:
        print("No complete snapshot found. Run download.py first.", file=sys.stderr)
        return 1

    latest_papers = None
    for position, (snapshot_dir, manifest) in enumerate(usable):
        is_latest = position == len(usable) - 1
        summary_path = summary_dir / f"{snapshot_dir.name}.json"
        if summary_path.exists() and not is_latest:
            continue
        papers, statuses = read_snapshot(snapshot_dir)
        write_json(summary_path, summarize(snapshot_dir.name, manifest["taken_at"], papers, statuses))
        print(f"Summarized {snapshot_dir.name}: {len(papers)} papers")
        if is_latest:
            latest_papers = papers

    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(summary_dir.glob("*.json"))]
    latest = summaries[-1]
    if latest["id"] != usable[-1][0].name:
        print(f"Raw snapshot {latest['id']} is missing, so its paper list cannot be rebuilt. "
              "Restore it or delete its summary.", file=sys.stderr)
        return 1
    first_ratings = summaries[0]["ratings_by_paper"] if len(summaries) > 1 else {}

    primary_areas = sorted(list({p["primary_area"] for p in latest_papers}))

    papers_doc = {
        "columns": ["number", "forum", "title", "primary_area", "keywords", "ratings", "confidence_mean", "first_mean"],
        "rows": rows_for_page(latest_papers, first_ratings),
    }
    site = {
        "venue": usable[-1][1]["venue"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest": latest["id"],
        "fields": [f for f in FIELDS if f in latest["histograms"]],
        "primary_areas": primary_areas,
        "snapshots": [{k: v for k, v in s.items() if k != "ratings_by_paper"} for s in summaries],
    }
    check_output(site, papers_doc)
    write_json(data_dir / "site.json", site)
    write_json(data_dir / "papers.json", papers_doc)
    print(f"Wrote {data_dir}: {len(papers_doc['rows'])} papers, {len(primary_areas)} primary areas, {len(summaries)} snapshot(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Offline tests with synthetic data. They make no network calls."""
import gzip
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_site
import download
import import_notes

VENUE = "ICLR.cc/2027/Conference"
CANARY = "CANARYREVIEWTEXT"
ABSTRACT = "CANARYABSTRACT"
AUTHOR = "CANARYAUTHOR"
SAMPLE_AREA = "Generative Models"


def review(number, index, rng, rating=None, late=False):
    text = f"{CANARY} {number} {index}"
    return {
        "id": f"r{number}x{index}",
        "forum": f"f{number}",
        "invitations": [f"{VENUE}/Submission{number}/-/Official_Review", f"{VENUE}/-/Edit"],
        "signatures": [f"{VENUE}/Submission{number}/Reviewer_CANARYSIG{index}"],
        "content": {
            "paper_summary": {"value": text},
            "critical_strengths": {"value": text},
            "critical_weaknesses": {"value": text},
            "questions_for_authors": {"value": text},
            "rating": {"value": rating if rating is not None else rng.choice([2, 4, 6, 8])},
            "confidence": {"value": rng.randint(1, 5)},
            "soundness": {"value": rng.randint(1, 4)},
            "presentation": {"value": rng.randint(1, 4)},
            "informativeness": {"value": rng.randint(1, 4)},
        },
    }


def submission(number, rng, venue_suffix="Submission", n_reviews=3, primary_area=SAMPLE_AREA):
    replies = [review(number, i, rng) for i in range(n_reviews)]
    replies.append({  # a non-review reply must be ignored
        "id": f"c{number}", "forum": f"f{number}",
        "invitations": [f"{VENUE}/Submission{number}/-/Official_Comment"],
        "content": {"comment": {"value": CANARY}, "rating": {"value": 1}},
    })
    return {
        "id": f"f{number}", "forum": f"f{number}", "number": number,
        "content": {
            "title": {"value": f"Paper  title {number}"},
            "abstract": {"value": ABSTRACT},
            "authors": {"value": [AUTHOR]},
            "keywords": {"value": ["reasoning models", "deep learning"]},
            "primary_area": {"value": primary_area},
            "venueid": {"value": f"{VENUE}/{venue_suffix}"},
        },
        "details": {"directReplies": replies},
    }


class FakeNote:
    def __init__(self, raw):
        self.raw, self.number, self.details = raw, raw["number"], raw["details"]

    def to_json(self):
        return {k: v for k, v in self.raw.items() if k not in ("number", "details")}


class FakeClient:
    def __init__(self, raws, fail_at=None):
        self.raws, self.fail_at, self.calls = raws, fail_at, []

    def get_notes(self, invitation, details, sort, limit, offset=None, with_count=None):
        assert invitation == f"{VENUE}/-/Submission" and details == "directReplies"
        self.calls.append(offset)
        if self.fail_at is not None and (offset or 0) >= self.fail_at:
            raise RuntimeError("{'name': 'ChallengeRequiredError', 'status': 403}")
        start = offset or 0
        notes = [FakeNote(r) for r in self.raws[start:start + limit]]
        return (notes, len(self.raws)) if with_count and offset is None else notes


def run_download(out, client, *extra):
    return download.main(["--out", str(out), "--page-size", "10", *extra],
                         client_factory=lambda: client, pause=0)


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out, self.docs = self.root / "snapshots", self.root / "docs"
        rng = random.Random(1)
        self.raws = [submission(n, rng) for n in range(1, 24)]
        self.raws.append(submission(24, rng, "Withdrawn_Submission"))
        self.raws.append(submission(25, rng, "Desk_Rejected_Submission", 0))
        self.raws.append(submission(26, rng, n_reviews=0))

    def tearDown(self):
        self.tmp.cleanup()

    def only_manifest(self):
        return json.loads(next(self.out.glob("*/manifest.json")).read_text(encoding="utf-8"))

    def test_one_request_per_page(self):
        client = FakeClient(self.raws)
        self.assertEqual(run_download(self.out, client), 0)
        self.assertEqual(client.calls, [None, 10, 20])
        manifest = self.only_manifest()
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["submissions"], 26)
        self.assertEqual(manifest["reviews"], 24 * 3)

    def test_resume_after_challenge_refetches_nothing(self):
        self.assertEqual(run_download(self.out, FakeClient(self.raws, fail_at=20)), 2)
        self.assertEqual(list(self.out.glob("*/manifest.json")), [])
        client = FakeClient(self.raws)
        self.assertEqual(run_download(self.out, client), 0)
        self.assertEqual(client.calls, [20])
        self.assertEqual(len(list(self.out.glob("*/"))), 1)
        self.assertEqual(self.only_manifest()["submissions"], 26)

    def test_missing_submissions_are_flagged(self):
        class Short(FakeClient):
            def get_notes(self, **kw):
                result = super().get_notes(**kw)
                return (result[0], result[1] + 5) if isinstance(result, tuple) else result
        self.assertEqual(run_download(self.out, Short(self.raws)), 1)
        self.assertFalse(self.only_manifest()["complete"])
        self.assertEqual(build_site.main(["--snapshots", str(self.out), "--docs", str(self.docs)]), 1)

    def build(self):
        self.assertEqual(build_site.main(["--snapshots", str(self.out), "--docs", str(self.docs)]), 0)
        return (json.loads((self.docs / "data/site.json").read_text(encoding="utf-8")),
                json.loads((self.docs / "data/papers.json").read_text(encoding="utf-8")))

    def test_page_data_has_no_private_text(self):
        run_download(self.out, FakeClient(self.raws))
        site, papers = self.build()
        blob = "".join(p.read_text(encoding="utf-8") for p in self.docs.rglob("*.json"))
        for banned in (CANARY, ABSTRACT, AUTHOR, "CANARYSIG", "Reviewer_", "Official_Review"):
            self.assertNotIn(banned, blob)
        raw = gzip.open(next(self.out.glob("*/page_0000000.json.gz")), "rt", encoding="utf-8").read()
        self.assertIn(CANARY, raw)  # the raw data did hold it

    def test_scores_are_correct(self):
        run_download(self.out, FakeClient(self.raws))
        site, papers = self.build()
        self.assertEqual(papers["columns"],
                         ["number", "forum", "title", "primary_area", "keywords", "ratings", "confidence_mean", "first_mean"])
        rows = {r[0]: r for r in papers["rows"]}
        self.assertNotIn(24, rows)  # withdrawn
        self.assertNotIn(25, rows)  # desk rejected
        self.assertEqual(rows[26][5], [])  # active, not reviewed yet
        self.assertEqual(rows[1][1], "f1")
        self.assertEqual(rows[1][2], "Paper title 1")
        self.assertEqual(rows[1][3], SAMPLE_AREA)
        self.assertEqual(rows[1][4], ["reasoning models", "deep learning"])
        expected = sorted((r["content"]["rating"]["value"]
                           for r in self.raws[0]["details"]["directReplies"]
                           if r["id"].startswith("r")), reverse=True)
        self.assertEqual(rows[1][5], expected)  # the comment's rating is not counted
        self.assertIn(SAMPLE_AREA, site["primary_areas"])
        latest = site["snapshots"][-1]
        self.assertEqual(latest["active"], 24)
        self.assertEqual(latest["withdrawn"], 1)
        self.assertEqual(latest["desk_rejected"], 1)
        self.assertEqual(latest["reviews"], 23 * 3)
        self.assertEqual(sum(c for _, c in latest["histograms"]["rating"]), 69)
        self.assertEqual(sum(c for _, c in latest["mean_distribution"]), 23)

    def test_second_snapshot_tracks_change(self):
        run_download(self.out, FakeClient(self.raws))
        site, papers = self.build()
        first_row = [r for r in papers["rows"] if r[0] == 1][0]
        first_mean = sum(first_row[5]) / len(first_row[5])
        for r in self.raws[0]["details"]["directReplies"]:
            if r["id"].startswith("r"):
                r["content"]["rating"]["value"] = 10
        first_dir = next(self.out.glob("*/manifest.json")).parent
        first_dir.rename(self.out / "2026-01-01T0000Z")
        run_download(self.out, FakeClient(self.raws), "--new")
        site, papers = self.build()
        self.assertEqual(len(site["snapshots"]), 2)
        row = [r for r in papers["rows"] if r[0] == 1][0]
        self.assertEqual(row[5], [10, 10, 10])
        self.assertAlmostEqual(row[7], first_mean, places=3)

    def save_pages(self, pages):
        folder = self.root / "notes"
        folder.mkdir()
        for index, raws in enumerate(pages):
            (folder / f"notes{index}.json").write_text(
                json.dumps({"notes": raws, "count": len(self.raws)}), encoding="utf-8")
        return ["--notes", str(folder), "--out", str(self.out)]

    def test_import_complete_pages(self):
        args = self.save_pages([self.raws[:10], self.raws[10:20], self.raws[20:]])
        self.assertEqual(import_notes.main(args), 0)
        self.assertTrue(self.only_manifest()["complete"])
        self.assertEqual(self.build()[0]["snapshots"][-1]["active"], 24)

    def test_import_refuses_duplicate_page(self):
        args = self.save_pages([self.raws[:10], self.raws[:10], self.raws[20:]])
        self.assertEqual(import_notes.main(args), 1)
        self.assertEqual(list(self.out.glob("*/manifest.json")), [])

    def test_text_scores_parse(self):
        self.assertEqual(build_site.to_number("6: marginally above"), 6)
        self.assertEqual(build_site.to_number(2.5), 2.5)
        self.assertIsNone(build_site.to_number("n/a"))
        self.assertIsNone(build_site.to_number(None))

    def test_whitespace_area_and_keyword_formats(self):
        self.assertEqual(build_site.sanitize_text("   "), "")
        self.assertEqual(build_site.sanitize_text(None), "")
        self.assertEqual(build_site.extract_keywords("deep learning, reinforcement learning"),
                         ["deep learning", "reinforcement learning"])
        self.assertEqual(build_site.extract_keywords(["nlp", "vision"]), ["nlp", "vision"])
        self.assertEqual(build_site.extract_keywords(""), [])


if __name__ == "__main__":
    unittest.main()

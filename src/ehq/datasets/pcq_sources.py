"""Deterministic fact extraction from structured PCQ source snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..hashing import sha256_file


_NUMBER = r"(?:[−–-]?\d+(?:\.\d+)?|…)"
_WEO_ROW = re.compile(
    rf"^\s*(?P<label>.+?)\s+"
    rf"(?P<v1>{_NUMBER})\s+(?P<v2>{_NUMBER})\s+"
    rf"(?P<v3>{_NUMBER})\s+(?P<v4>{_NUMBER})\s+"
    rf"(?P<v5>{_NUMBER})\s+(?P<v6>{_NUMBER})\s+"
    rf"(?P<v7>{_NUMBER})\s+(?P<v8>{_NUMBER})\s+"
    rf"(?P<v9>{_NUMBER})\s*$"
)


def _localized(value: Any) -> str:
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, Mapping) and entry.get("Description"):
                return str(entry["Description"])
    return ""


def _iso_date_from_epoch_ms(value: Any) -> str:
    return datetime.fromtimestamp(
        int(value) / 1000,
        tz=timezone.utc,
    ).date().isoformat()


def evidence_record(
    source: Mapping[str, Any],
    *,
    project_root: Path,
    evidence: str,
    locator: Mapping[str, Any],
) -> Dict[str, Any]:
    snapshot_path = str(source["snapshot_path"])
    return {
        "source_id": str(source["source_id"]),
        "url": str(source["url"]),
        "publisher": str(source["publisher"]),
        "title": str(source["title"]),
        "published_at": str(source["published_at"]),
        "retrieved_at": str(source.get("retrieved_at") or "2026-07-31"),
        "verified_at": "2026-07-31",
        "snapshot_path": snapshot_path,
        "snapshot_sha256": sha256_file(project_root / snapshot_path),
        "locator": dict(locator),
        "evidence": evidence,
    }


def temporal_novelty_record(
    source: Mapping[str, Any],
    *,
    claim_became_true_at: str,
    basis: str,
    evidence: str,
) -> Dict[str, Any]:
    """Create a source-level attestation; release still requires human review."""

    return {
        "status": "source-verified-post-cutoff",
        "basis": basis,
        "claim_became_true_at": claim_became_true_at,
        "source_id": str(source["source_id"]),
        "reviewer_type": "deterministic-structured-source-parser",
        "verified_at": "2026-07-31",
        "evidence": evidence,
    }


def build_sports_facts(
    source: Mapping[str, Any],
    *,
    project_root: Path,
) -> List[Dict[str, Any]]:
    snapshot = project_root / str(source["snapshot_path"])
    payload = json.loads(snapshot.read_text(encoding="utf-8-sig"))
    matches = payload.get("Results")
    if not isinstance(matches, list) or len(matches) != 104:
        raise ValueError(f"Expected 104 FIFA matches, found {len(matches or [])}")
    matches = sorted(matches, key=lambda row: int(row["MatchNumber"]))
    facts: List[Dict[str, Any]] = []
    for match in matches:
        number = int(match["MatchNumber"])
        home = _localized(match["Home"]["TeamName"])
        away = _localized(match["Away"]["TeamName"])
        home_score = int(match["HomeTeamScore"])
        away_score = int(match["AwayTeamScore"])
        event_date = str(match["Date"])[:10]
        score = f"{home_score}-{away_score}"
        evidence = (
            f"FIFA match {number}: {home} {score} {away}; "
            f"played on {event_date}."
        )
        facts.append(
            {
                "subcategory": "PCQ-SPO",
                "event_date": event_date,
                "question_target": (
                    f"the score reported by FIFA for 2026 World Cup match "
                    f"{number} between {home} and {away}"
                ),
                "gold_answer": score,
                "acceptable_answers": [
                    f"{home} {score} {away}",
                    f"{home_score} to {away_score}",
                ],
                "temporal_novelty": temporal_novelty_record(
                    source,
                    claim_became_true_at=event_date,
                    basis="official_result",
                    evidence=evidence,
                ),
                "source_evidence": [
                    evidence_record(
                        source,
                        project_root=project_root,
                        evidence=evidence,
                        locator={
                            "json_path": (
                                f"Results[MatchNumber={number}]."
                                "HomeTeamScore/AwayTeamScore"
                            ),
                            "match_id": match.get("IdMatch"),
                        },
                    )
                ],
            }
        )
    for match in matches[:46]:
        number = int(match["MatchNumber"])
        home = _localized(match["Home"]["TeamName"])
        away = _localized(match["Away"]["TeamName"])
        attendance = int(match["Attendance"])
        event_date = str(match["Date"])[:10]
        evidence = (
            f"FIFA recorded attendance of {attendance:,} for match {number}, "
            f"{home} versus {away}, on {event_date}."
        )
        facts.append(
            {
                "subcategory": "PCQ-SPO",
                "event_date": event_date,
                "question_target": (
                    f"FIFA's recorded attendance for 2026 World Cup match "
                    f"{number} between {home} and {away}"
                ),
                "gold_answer": f"{attendance:,}",
                "acceptable_answers": [str(attendance)],
                "temporal_novelty": temporal_novelty_record(
                    source,
                    claim_became_true_at=event_date,
                    basis="official_result",
                    evidence=evidence,
                ),
                "source_evidence": [
                    evidence_record(
                        source,
                        project_root=project_root,
                        evidence=evidence,
                        locator={
                            "json_path": f"Results[MatchNumber={number}].Attendance",
                            "match_id": match.get("IdMatch"),
                        },
                    )
                ],
            }
        )
    if len(facts) != 150:
        raise AssertionError(f"Sports extraction produced {len(facts)} facts")
    return facts


def parse_weo_table_rows(layout_text: str) -> List[Dict[str, Any]]:
    rows = []
    for line in layout_text.splitlines():
        match = _WEO_ROW.fullmatch(line)
        if not match:
            continue
        label = re.sub(r"\s+\d+/$", "", match.group("label")).strip()
        if not label:
            continue
        values = [
            match.group(f"v{index}").replace("−", "-").replace("–", "-")
            for index in range(1, 10)
        ]
        rows.append({"label": label, "values": values, "source_line": line.strip()})
    return rows


def build_economics_facts(
    source: Mapping[str, Any],
    *,
    project_root: Path,
) -> List[Dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "PCQ-ECO extraction requires the optional 'sources' dependencies"
        ) from exc
    snapshot = project_root / str(source["snapshot_path"])
    with pdfplumber.open(snapshot) as pdf:
        page_number = int(source.get("page") or 13)
        page = pdf.pages[page_number - 1]
        layout_text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=2)
    rows = parse_weo_table_rows(layout_text or "")
    if len(rows) != 40:
        raise ValueError(f"Expected 40 IMF table rows, found {len(rows)}")
    metrics = (
        ("2026 year-over-year projection", 2),
        ("2027 year-over-year projection", 3),
        ("2026 difference from the April 2026 WEO projection", 4),
        ("2027 difference from the April 2026 WEO projection", 5),
    )
    facts: List[Dict[str, Any]] = []
    for row in rows:
        for metric, value_index in metrics:
            value = row["values"][value_index]
            if value == "…":
                continue
            evidence = (
                f"IMF Table 1 row '{row['label']}' reports {metric}: "
                f"{value} percent."
            )
            facts.append(
                {
                    "subcategory": "PCQ-ECO",
                    "event_date": str(source["published_at"]),
                    "question_target": (
                        f"the {metric} for {row['label']} in the IMF's "
                        "July 2026 World Economic Outlook Update"
                    ),
                    "gold_answer": f"{value}%",
                    "acceptable_answers": [f"{value} percent", value],
                    "temporal_novelty": temporal_novelty_record(
                        source,
                        claim_became_true_at=str(source["published_at"]),
                        basis="new_forecast",
                        evidence=evidence,
                    ),
                    "source_evidence": [
                        evidence_record(
                            source,
                            project_root=project_root,
                            evidence=evidence,
                            locator={
                                "page": int(source.get("page") or 13),
                                "table": "Table 1",
                                "row": row["label"],
                                "metric": metric,
                                "extracted_row": row["source_line"],
                            },
                        )
                    ],
                }
            )
    if len(facts) < 150:
        raise ValueError(f"Economics extraction produced only {len(facts)} facts")
    return facts[:150]


def build_world_facts(
    source: Mapping[str, Any],
    *,
    project_root: Path,
) -> List[Dict[str, Any]]:
    snapshot = project_root / str(source["snapshot_path"])
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    events = payload.get("features")
    if not isinstance(events, list) or len(events) != 50:
        raise ValueError(f"Expected 50 USGS events, found {len(events or [])}")
    facts: List[Dict[str, Any]] = []
    for event in events:
        properties = event["properties"]
        event_id = str(event["id"])
        event_date = _iso_date_from_epoch_ms(properties["time"])
        magnitude = str(properties["mag"])
        place = str(properties["place"])
        depth = str(event["geometry"]["coordinates"][2])
        base_locator = {"event_id": event_id}
        fact_specs = (
            (
                f"the magnitude USGS reported for reviewed event {event_id} "
                f"near {place}",
                magnitude,
                [f"M {magnitude}", f"magnitude {magnitude}"],
                f"USGS event {event_id} had magnitude {magnitude} near {place}.",
                "properties.mag",
            ),
            (
                f"the location description USGS reported for reviewed event "
                f"{event_id} of magnitude {magnitude}",
                place,
                [],
                f"USGS described event {event_id} as occurring {place}.",
                "properties.place",
            ),
            (
                f"the hypocentral depth USGS reported for reviewed event "
                f"{event_id} near {place}",
                f"{depth} km",
                [depth, f"{depth} kilometres", f"{depth} kilometers"],
                f"USGS event {event_id} had a reported depth of {depth} km.",
                "geometry.coordinates[2]",
            ),
        )
        for target, gold, acceptable, evidence, json_path in fact_specs:
            facts.append(
                {
                    "subcategory": "PCQ-WOR",
                    "event_date": event_date,
                    "question_target": target,
                    "gold_answer": gold,
                    "acceptable_answers": acceptable,
                    "temporal_novelty": temporal_novelty_record(
                        source,
                        claim_became_true_at=event_date,
                        basis="event_occurrence",
                        evidence=evidence,
                    ),
                    "source_evidence": [
                        evidence_record(
                            source,
                            project_root=project_root,
                            evidence=evidence,
                            locator={**base_locator, "json_path": json_path},
                        )
                    ],
                }
            )
    if len(facts) != 150:
        raise AssertionError(f"World extraction produced {len(facts)} facts")
    return facts


def build_politics_programme_facts(
    source: Mapping[str, Any],
    *,
    project_root: Path,
) -> List[Dict[str, Any]]:
    try:
        from lxml import html
    except ImportError as exc:
        raise RuntimeError(
            "PCQ-POL programme extraction requires the optional 'sources' "
            "dependencies"
        ) from exc
    snapshot = project_root / str(source["snapshot_path"])
    root = html.fromstring(snapshot.read_bytes())
    panel_dates = {}
    for element in root.iter():
        controls = str(element.get("aria-controls") or "")
        if not controls or element.get("data-cmp-hook-tabs") != "tab":
            continue
        label = " ".join(" ".join(element.itertext()).split())
        match = re.search(r"\b([678])\s+Jul\b", label)
        if match:
            panel_dates[controls] = f"2026-07-{int(match.group(1)):02d}"
    activities = []
    seen = set()
    for block in root.iter("div"):
        if "activity-item-block" not in (block.get("class") or ""):
            continue
        hour_elements = [
            element
            for element in block.iter("span")
            if "activity-hour" in (element.get("class") or "")
        ]
        title_elements = [
            element
            for element in block.iter("span")
            if "activity-title" == (element.get("class") or "").strip()
        ]
        if not hour_elements or not title_elements:
            continue
        hour = " ".join(" ".join(hour_elements[0].itertext()).split())
        title_element = title_elements[0]
        title_parts = []
        if title_element.text and title_element.text.strip():
            title_parts.append(title_element.text.strip())
        for child in title_element:
            if "activity-title-superindex" in (child.get("class") or ""):
                continue
            if "activity-subtitle" in (child.get("class") or ""):
                continue
            if child.tail and child.tail.strip():
                title_parts.append(child.tail.strip())
        title = " ".join(" ".join(title_parts).split())
        subtitle_elements = [
            element
            for element in title_element.iter("span")
            if "activity-subtitle" in (element.get("class") or "")
        ]
        location = ""
        if subtitle_elements:
            location = " ".join(
                " ".join(subtitle_elements[0].itertext()).split()
            )
            location = re.sub(r"^Location:\s*", "", location)
        event_date = ""
        ancestor = block
        while ancestor is not None:
            panel_id = str(ancestor.get("id") or "")
            if panel_id in panel_dates:
                event_date = panel_dates[panel_id]
                break
            ancestor = ancestor.getparent()
        if not event_date:
            for link in block.iter("a"):
                href = str(link.get("href") or "")
                match = re.search(r"/2026/07/(0[678])/", href)
                if match:
                    event_date = f"2026-07-{match.group(1)}"
                    break
        if not event_date:
            event_date = "2026-07-07/2026-07-08"
        identity = (event_date, hour, title, location)
        if not title or identity in seen:
            continue
        seen.add(identity)
        activities.append(
            {
                "event_date": event_date,
                "hour": hour,
                "title": title,
                "location": location,
            }
        )
    if len(activities) < 25:
        raise ValueError(
            f"Expected at least 25 unique NATO programme activities, "
            f"found {len(activities)}"
        )
    facts: List[Dict[str, Any]] = []
    for index, activity in enumerate(activities, 1):
        event_date = activity["event_date"]
        hour = activity["hour"]
        title = activity["title"]
        location = activity["location"]
        evidence = (
            f"NATO programme entry {index} on {event_date} at {hour}: {title}"
            + (f"; location: {location}." if location else ".")
        )
        common = {
            "subcategory": "PCQ-POL",
            "event_date": event_date,
            "temporal_novelty": temporal_novelty_record(
                source,
                claim_became_true_at=event_date,
                basis="new_schedule",
                evidence=evidence,
            ),
            "source_evidence": [
                evidence_record(
                    source,
                    project_root=project_root,
                    evidence=evidence,
                    locator={
                        "programme_entry": index,
                        "date": event_date,
                        "time": hour,
                    },
                )
            ],
        }
        facts.append(
            {
                **common,
                "question_target": (
                    f"the activity in NATO Ankara Summit programme entry "
                    f"{index}, scheduled for {event_date} at {hour}"
                ),
                "gold_answer": title,
                "acceptable_answers": [],
            }
        )
        facts.append(
            {
                **common,
                "question_target": (
                    f"the scheduled time for '{title}' in NATO Ankara Summit "
                    f"programme entry {index} on {event_date}"
                ),
                "gold_answer": hour,
                "acceptable_answers": [],
            }
        )
        if location:
            facts.append(
                {
                    **common,
                    "question_target": (
                        f"the listed location for '{title}' in NATO Ankara "
                        f"Summit programme entry {index}"
                    ),
                    "gold_answer": location,
                    "acceptable_answers": [],
                }
            )
    if len(facts) < 50:
        raise ValueError(
            f"NATO programme extraction produced only {len(facts)} facts"
        )
    return facts[:50]


def assign_fact_ids(
    facts: Iterable[Mapping[str, Any]],
    *,
    subcategory: str,
) -> List[Dict[str, Any]]:
    result = []
    for index, fact in enumerate(facts, 1):
        result.append(
            {
                "fact_id": f"{subcategory}-{index:04d}",
                **dict(fact),
                "verification": {
                    "status": "source-verified",
                    "reviewer_type": "automated-source-check",
                    "verified_at": "2026-07-31",
                },
            }
        )
    return result


def source_by_parser(
    manifest: Mapping[str, Any],
    parser: str,
) -> Mapping[str, Any]:
    matches = [
        source
        for source in manifest.get("sources") or []
        if isinstance(source, Mapping) and source.get("parser") == parser
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one source for parser {parser!r}, found {len(matches)}")
    return {**matches[0], "retrieved_at": manifest.get("retrieved_at")}


def verify_manifest_snapshots(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
) -> Dict[str, str]:
    hashes = {}
    for source in manifest.get("sources") or []:
        if not isinstance(source, Mapping):
            raise ValueError("Manifest sources must be objects")
        path = project_root / str(source["snapshot_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing PCQ source snapshot: {path}")
        hashes[str(source["source_id"])] = sha256_file(path)
    return hashes

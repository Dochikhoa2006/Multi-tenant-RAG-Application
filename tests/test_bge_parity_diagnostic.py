"""Offline validation of parity accounting; no model or database access."""
import copy
import json
from pathlib import Path

import pytest

from scripts.diagnose_bge_parity import comparison, decode_candidates, write_new


def test_comparison_reports_real_inversions_ties_and_floor_equality():
    rows = [
        dict(candidate_id="a", captured_logit=-1.1, floor=-1.0, ref=-1.0),
        dict(candidate_id="b", captured_logit=-0.5, floor=-1.0, ref=-2.0),
        dict(candidate_id="c", captured_logit=-0.5, floor=-1.0, ref=-2.1),
    ]
    result = comparison(rows, "ref")
    assert result["floor_flip_ids"] == ["a", "b", "c"]
    assert result["strict_pairwise_inversions"] == 2
    assert result["tie_status_changes"] == 1
    assert result["max_rank_displacement"] == 2
    assert rows[0]["ref_floor_pass"] is True
    assert result["max_abs_logit_error"] == pytest.approx(1.6)


def operation_fixture():
    fields = "candidate_id|hybrid_rank|bge_rank|raw_score_hex|floor_hex|floor_pass"
    rows = [f"00000000-0000-0000-0000-{i:012d}|{i}|{i}|-0x1p+1|-0x1p+0|0" for i in range(1,41)]
    samples = {}
    for page, items in enumerate((rows[:32], rows[32:]),1):
        for kind in ("decisions", "text_digests"):
            values = items if kind == "decisions" else [r.split('|')[0]+":"+"0"*64 for r in items]
            samples[f"policy_candidate_{kind}_page_{page}"] = dict(items=values,exact_count=len(values),truncated=False)
    return dict(texts={"policy_candidate_decision_fields":fields},samples=samples)


def test_capture_completeness_and_order():
    op = operation_fixture()
    rows = decode_candidates(op,"policy")
    assert len(rows)==40
    assert [r['hybrid_rank'] for r in rows] == list(range(1,41))
    assert all(r['captured_logit']==-2.0 for r in rows)


@pytest.mark.parametrize("corruption", ["truncated", "duplicate", "incomplete", "floor"])
def test_capture_fails_closed(corruption):
    op = copy.deepcopy(operation_fixture())
    page=op['samples']['policy_candidate_decisions_page_1']
    if corruption=='truncated': page['truncated']=True
    if corruption=='duplicate': page['items'][1]=page['items'][0]
    if corruption=='incomplete': page['items'].pop()
    if corruption=='floor': page['items'][0]=page['items'][0][:-1]+'1'
    with pytest.raises(ValueError): decode_candidates(op,'policy')


def test_output_never_overwrites_prior_evidence(tmp_path: Path):
    path=tmp_path/'report.json'
    write_new(path,{'original':True})
    with pytest.raises(FileExistsError): write_new(path,{'replacement':True})
    assert json.loads(path.read_text())=={'original':True}


def test_nonfinite_comparison_rejected():
    with pytest.raises(ValueError):
        comparison([dict(candidate_id='a',captured_logit=0.,floor=-1.,ref=float('nan'))],'ref')

"""SCP-ECG code → cardiac state class mapping for Modèle A.

Three-class taxonomy validated up-front:

    0 = normal     — sinus rhythm / NORM only, no significant codes
    1 = suspect    — chronic markers / HF substrate / non-acute anomalies
    2 = critique   — acute or high-risk arrhythmias / blocks / ischemia

Multi-label resolution rule: severity-max
    if any code ∈ CRITICAL_CODES → class 2
    elif any code ∈ SUSPECT_CODES → class 1
    else → class 0

The mapping is intentionally explicit: any future change is a single-file
review, and unmapped codes default to "suspect" (conservative) with a
warning emitted on first encounter.
"""

from __future__ import annotations

import ast
import logging
from typing import Iterable, Mapping

from .config import CRITICAL_CLASS, NORMAL_CLASS, SUSPECT_CLASS


logger = logging.getLogger(__name__)


NORMAL_CODES: frozenset[str] = frozenset({
    "NORM",
    "SR",
    "SARRH",
    "SVARR",
})

CRITICAL_CODES: frozenset[str] = frozenset({
    "AFIB",
    "AFLT",
    "PSVT",
    "SVTAC",
    "VT",
    "BIGU",
    "TRIGU",
    "STACH",
    "2AVB",
    "3AVB",
    "WPW",
    "AVB",
})

SUSPECT_CODES: frozenset[str] = frozenset({
    "LVH",
    "RVH",
    "VCLVH",
    "LAO/LAE",
    "RAO/RAE",
    "LAFB/LPFB",
    "LAFB",
    "LPFB",
    "IRBBB",
    "CRBBB",
    "CLBBB",
    "ILBBB",
    "IVCD",
    "1AVB",
    "IMI",
    "ASMI",
    "AMI",
    "ALMI",
    "ILMI",
    "IPLMI",
    "IPMI",
    "INJAS",
    "INJAL",
    "INJIN",
    "INJLA",
    "INJIL",
    "ISCA",
    "ISCAS",
    "ISCAL",
    "ISCAN",
    "ISCIN",
    "ISCIL",
    "ISCLA",
    "ISC_",
    "QWAVE",
    "NDT",
    "NST_",
    "LOWT",
    "STD_",
    "STE_",
    "ABQRS",
    "NT_",
    "DIG",
    "EL",
    "PAC",
    "PVC",
    "SBRAD",
    "PACE",
    "LPR",
    "LVOLT",
    "HVOLT",
    "TAB_",
    "INVT",
    "PRC(S)",
    "LNGQT",
    "QAB",
    "LMI",
    "AMIs",
    "SEHYP",
    "RVOLT",
})


_unmapped_warned: set[str] = set()


def assign_class(scp_codes: Mapping[str, float] | None) -> int:
    """Return the 3-class label for a record's ``scp_codes`` dictionary.

    Parameters
    ----------
    scp_codes:
        Dict mapping SCP code → likelihood %. Likelihood values are not
        used for class assignment — presence alone matters, matching the
        PTB-XL multi-label convention.

    Returns
    -------
    int
        0 normal, 1 suspect, 2 critique. Empty or None input → 0.
    """
    if not scp_codes:
        return NORMAL_CLASS
    codes = set(scp_codes.keys())
    if codes & CRITICAL_CODES:
        return CRITICAL_CLASS
    has_normal = bool(codes & NORMAL_CODES)
    other = codes - NORMAL_CODES
    if not other:
        return NORMAL_CLASS if has_normal else NORMAL_CLASS
    if other & SUSPECT_CODES:
        return SUSPECT_CLASS
    unmapped = other - SUSPECT_CODES - NORMAL_CODES - CRITICAL_CODES
    new = unmapped - _unmapped_warned
    if new:
        for code in new:
            logger.warning("SCP code %r unmapped — defaulting to suspect", code)
        _unmapped_warned.update(new)
    return SUSPECT_CLASS


def parse_scp_codes(raw: str | dict | None) -> dict[str, float]:
    """Parse the ``scp_codes`` column from CSV (always stored as a string).

    Returns an empty dict on parse failure.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def assign_class_from_raw(raw: str | dict | None) -> int:
    return assign_class(parse_scp_codes(raw))


def class_distribution(scp_series: Iterable) -> dict[str, int]:
    """Return ``{normal, suspect, critique}`` counts for a Series of raw cells."""
    counts = {"normal": 0, "suspect": 0, "critique": 0}
    names = {0: "normal", 1: "suspect", 2: "critique"}
    for raw in scp_series:
        counts[names[assign_class_from_raw(raw)]] += 1
    return counts


def _self_test() -> None:
    """Light unit tests — run with ``python -m scripts.modelA.label_mapping``."""
    cases = [
        ({"NORM": 100.0, "SR": 0.0}, NORMAL_CLASS, "norm+sr"),
        ({"SR": 100.0}, NORMAL_CLASS, "sr only"),
        ({"NORM": 100.0, "LVOLT": 0.0}, SUSPECT_CLASS, "norm+lvolt"),
        ({"LVH": 100.0, "SR": 0.0}, SUSPECT_CLASS, "lvh"),
        ({"CLBBB": 100.0}, SUSPECT_CLASS, "clbbb"),
        ({"AFIB": 100.0, "SR": 0.0}, CRITICAL_CLASS, "afib"),
        ({"STACH": 100.0, "NORM": 0.0}, CRITICAL_CLASS, "stach"),
        ({"WPW": 100.0}, CRITICAL_CLASS, "wpw"),
        ({"AFIB": 100.0, "LVH": 100.0}, CRITICAL_CLASS, "afib+lvh -> critical"),
        ({}, NORMAL_CLASS, "empty"),
        (None, NORMAL_CLASS, "none"),
        ("{'AFIB': 100.0}", CRITICAL_CLASS, "raw string"),
    ]
    for inp, expected, name in cases:
        got = (
            assign_class_from_raw(inp)
            if isinstance(inp, str) or inp is None
            else assign_class(inp)
        )
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] {name:<30} expected={expected} got={got}")
        assert got == expected, f"{name}: expected {expected}, got {got}"
    print("All label_mapping tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()

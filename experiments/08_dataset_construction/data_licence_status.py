"""Reviewed source-status labels shared by the dataset builders.

These describe the checked distributions; they do not certify rights to
underlying material. See DATA_PROVENANCE.md at the project root.
"""

LICENCE_STATUS = {
    "CIDAR": "Apache 2.0; research-purpose wording",
    "EASC": "Terms unresolved for the sampled mirror",
    "ARCD": "MIT declared; incomplete data terms",
}

LICENCE_REVIEW = {
    "reviewed": "2026-11-01",
    "provenance": "DATA_PROVENANCE.md",
    "scope": "Licence metadata correction only; prompts, IDs, splits and "
             "experimental results are unchanged. Checked source revisions "
             "are review snapshots, not recovered original sampling pins.",
}

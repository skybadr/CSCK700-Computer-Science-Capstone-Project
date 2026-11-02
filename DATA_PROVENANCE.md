# Data provenance and licence status review

Reviewed 1 November 2026. This corrects the source-status labels in the
dissertation's Table 4, repository documentation and the v2 and fresh-exam
dataset files. It changes metadata only. Prompt text, source IDs, record
IDs, task categories, splits, gold answers and stored experimental outputs
remain unchanged.

## Checked source distributions

| Source sampled by the builders | Reviewed status | Review snapshot |
|---|---|---|
| `arbml/CIDAR` | Apache 2.0; research-purpose wording | [`bc2f9d7a9de34b534b126c32b8cce98e098138df`](https://huggingface.co/datasets/arbml/CIDAR/blob/bc2f9d7a9de34b534b126c32b8cce98e098138df/README.md) |
| `arbml/easc` (Hub name `arbml/EASC`) | Terms unresolved for the sampled mirror | [`fbc3e5649c51cfd60e2a9d1b8a730ec4f46412a1`](https://huggingface.co/datasets/arbml/EASC/tree/fbc3e5649c51cfd60e2a9d1b8a730ec4f46412a1) |
| `hsseinmz/arcd` | MIT declared; incomplete data terms | [`cc6906b6eda547e4ffc63b8d88ccca7e0515187a`](https://huggingface.co/datasets/hsseinmz/arcd/blob/cc6906b6eda547e4ffc63b8d88ccca7e0515187a/README.md) |

CIDAR's card declares Apache 2.0 in its metadata and licence section, but
also describes the dataset as intended for research purposes. Both are
recorded here; the earlier CC BY-NC 4.0 label was not substantiated.

The sampled EASC mirror has no dataset card or licence declaration in the
checked revision. The author-hosted EASC distribution is a different
package; its terms cannot simply be assigned to the sampled mirror. The
earlier label `research-use` did not identify verified, version-specific
terms, so the current label records the unresolved status.

ARCD's card declares MIT in its front matter, while its Licensing
Information section is uncompleted. Its contexts originate in Wikipedia.
The metadata declaration alone does not resolve the terms applicable to
all underlying material; the earlier CC BY-SA 4.0 label was not
substantiated for this distribution.

These commits pin the sources checked during this review. The original
sampling scripts did not record immutable revisions, and these are not
claimed to be the original sampled snapshots. The historical construction
records and approved proposal/design-report images are retained as such.
Unresolved provenance and permissions require confirmation from the data
owners before redistribution or reuse beyond the permitted scope.

## Revision and reproducibility

The correction affects 441 v2 rows (250 CIDAR, 100 EASC, 91 ARCD) and 160
exam rows (84 CIDAR, 40 EASC, 36 ARCD). Each affected `license` field now
uses the same reviewed label as Table 4. The datasets' `licence_review`
metadata identifies this review without changing their experimental
version or sample composition. The shared labels in
`experiments/08_dataset_construction/data_licence_status.py` are applied by
all three builders, including assembly from the retained historical raw
construction record.

File hashes necessarily change when metadata changes. The before/after
hashes and hashes of prompt records with only the `license` field excluded
are recorded in
[`experiments/AUDIT/licence_metadata_revision.json`](experiments/AUDIT/licence_metadata_revision.json).
Stored experimental outputs have not been regenerated.

## AraPromptBench v3 combined release (2 November 2026)

`AraPromptBench_v3.json` contains the 1,000 original v2 records followed
by the 400 fresh-exam records. It preserves every record's source,
source ID, licence label, prompt, ID, split and any gold answers.
The original 800 dev / 200 test / 400 exam partitions remain distinct.
The component hashes and complete component metadata are included in the
combined metadata. The pilot and synthetic sensitivity pool remain separate.

| Source | Combined records | Inherited licence/status label |
|---|---:|---|
| CIDAR | 334 | Apache 2.0; research-purpose wording |
| Aya | 156 | Apache 2.0 |
| XL-Sum | 210 | CC BY-NC-SA 4.0 |
| EASC | 140 | Terms unresolved for the sampled mirror |
| TyDi-QA | 223 | Apache 2.0 |
| ARCD | 127 | MIT declared; incomplete data terms |
| Synthetic creative prompts | 210 | CC BY 4.0 |

These are the component datasets' reviewed labels, inherited without a
new source-rights review. Combining the records does not resolve EASC's
or ARCD's outstanding terms, remove source-specific conditions, or assign
a blanket licence to v3. The APCS package's MIT licence covers its code
and is not a licence for third-party dataset content.

Synthetic provenance remains component-specific: the 150 v2 creative
briefs used Claude Fable 5 (Anthropic), while the 60 exam briefs used
Claude Opus 5.5 (Anthropic) on 4 October 2026. V3 metadata records both
generators and their construction files.

Rebuild the combined file with
`python experiments/08_dataset_construction/assemble_v3.py`.
Use the same command with `--check` to verify the saved release against the
unchanged component files. No experimental output was regenerated.

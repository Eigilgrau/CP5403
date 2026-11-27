# Preserved Development Archive

These are historical launchers and non-identical sync-conflict copies, not the recommended reproduction entry points. Start with [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

- `notebooks/validation-pilots`: earlier C/D prompt and generation attempts.
- `notebooks/exploratory-judges`: calibration and semantic-evaluation attempts.
- `notebooks/post-test-development`: experiments after the frozen result; they do not replace it.
- `notebooks/literature-prototypes`: earlier ingestion/chunking launchers.
- `sync-conflicts`: files formerly named `... (1)...`. They differ from the canonical files and were preserved rather than discarded as duplicates.
- `guides`: original root guides before the reviewer navigation update.

`relocation_manifest.json` records the original path, archived path and SHA-256 for every moved file. File contents were not edited. `pre_cleanup_hashes.json` records the pre-cleanup module/config/notebook fingerprints for verification. To reverse a relocation, move that recorded archived file back to its recorded original path only if the destination is absent and its SHA-256 matches; do not overwrite a newer file.

Archived notebook paths do not redefine the project root. Most launchers explicitly change into CP5403. Some notebooks pin older source hashes; those guards may intentionally reject the currently frozen modules. Consult the associated run record/snapshot before attempting historical execution. No historical result or failed attempt was removed.

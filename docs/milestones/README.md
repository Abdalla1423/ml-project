# Milestones

Each milestone gets its own folder `msN/` with the same four files:

| File | Content |
|---|---|
| `sheet.pdf` | The official milestone sheet (copied in by hand). |
| `requirements.md` | Every requirement from the sheet as a checklist item (`- [ ]`) with an acceptance criterion. Used to verify the submission. |
| `plan.md` | Team split for this milestone and dependencies between people. Personal plans stay in each person's gitignored `CLAUDE.local.md`. |
| `status.md` | Progress log, updated as work is done. |

## Starting a new milestone

1. Create `docs/milestones/msN/` with the four files above. Copy the sheet in as `sheet.pdf`.
2. Write `requirements.md` faithfully from the sheet before planning.
3. Update the "Current milestone" line at the top of `CLAUDE.md` and the reading list below it.
4. Update the status table below.
5. Move any decisions from the previous milestone that are still open into the new context
   (they stay in `docs/decisions.md`, no need to copy).

## Submitting a milestone

1. Go through `requirements.md` and tick every item, with a reference to the test or file that proves it.
2. Tag the submitted commit: `git tag msN && git push origin msN`.
3. Make sure `marvinsxtr` has access to the repo.
4. Set the status below to `submitted`.

## Status

| Milestone | Topic | Status |
|---|---|---|
| ms1 | Data and Prototype | in progress |

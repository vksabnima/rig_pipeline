# Annotation task — friends brief

Thanks for helping. We need clean human-validated labels on regulatory text for a research paper. Three separate worksheets — total work is roughly 6-7 hours if one person does all three, much less if split.

## What you're being asked to do

Read short passages of financial/legal regulation text and fill in six structured columns per passage. It's read-then-type; no research, no looking things up. **Quality matters more than speed.**

## Before you start

Read `ANNOTATION_GUIDE.md` (in this folder). It has the field definitions and three fully worked examples. Keep it open while you work.

**Critical step** for any worksheet that has `claude_*` columns visible: **hide those columns** before you start (Excel: select the columns → right-click → Hide). You need to form your own answer first. Looking at the AI's answer biases yours.

## The three worksheets

### Worksheet 1: `kappa_300_worksheet.csv` — Basel III (most important)
- 300 rows, roughly 3.5–4 hours
- Basel III banking regulation text
- Each row has the AI's proposed annotation in `claude_*` columns (hide them) and blank `human_*` columns for you to fill
- This expands a smaller pilot and is the highest-priority deliverable

### Worksheet 2: `regbi_gold_worksheet.csv` — SEC Regulation Best Interest
- 100 rows, roughly 75 minutes
- US broker-dealer conduct regulation
- No AI annotations shown — you're creating a fresh independent gold standard
- Only `human_*` columns to fill (plus `notes`)

### Worksheet 3: `gdpr_human_worksheet.csv` — GDPR (EU privacy)
- 50 rows, roughly 40 minutes
- EU data protection regulation
- Fresh independent annotation like Worksheet 2
- Same six human columns to fill

## Can this be split across multiple people?

Yes. Ideal: **3 annotators each doing all 3 worksheets** (gives inter-rater agreement). Acceptable: **3 people each taking one worksheet**. Minimum: one person doing all three.

If multiple people annotate the SAME worksheet, save their copies with different filenames: `kappa_300_alice.csv`, `kappa_300_bob.csv`, etc. That lets us compute agreement between annotators, which is very valuable for the paper.

## Payment / acknowledgement

- Payment expectation from Vikash: [he'll confirm]
- For acknowledgement in the paper (if you want it): give him your name and preferred affiliation

## When you're done

Save each file as CSV (UTF-8 encoding). **Filename must stay the same** — don't rename. Send them back to Vikash.

## Questions?

Ask Vikash **before** you start, not halfway through. Common things to clarify:
- Any field definition that confuses you after reading the guide
- What to do if a clause is purely descriptive (answer: leave obligation blank)
- Whether to paraphrase or lift text verbatim (answer: lift verbatim)

## One-page cheat sheet

| Column | What goes in it | Example |
|---|---|---|
| `human_subject` | Who is regulated | `bank`, `broker-dealer`, `controller` |
| `human_obligation` | What they must do (verb phrase) | `calculate credit amounts`, `disclose conflicts` |
| `human_condition` | When/where it applies | `for OTC derivatives`, `to retail customers` |
| `human_threshold` | Any number or quantity | `10%`, `USD 5 million`, `150 basis points` |
| `human_deadline` | Any time limit | `within 72 hours`, `by 2025`, `prior to engagement` |
| `human_exception` | Any unless/except clause | `unless authorised`, `except small firms` |

**If it's not literally in the text, leave the column blank.**

# Regulatory Obligation Annotation Guide

**Read this before you start. Keep it open while annotating.**

You are extracting structured information from legal regulatory text.
For each clause you will fill in six fields using short phrases lifted
from the source text.

---

## Before you start

1. You will receive one CSV file. Open it in Excel, Google Sheets, LibreOffice, or a CSV editor.
2. In the CSV, there is a column called `source_text` — this is the clause to read.
3. For each row, you fill in six columns: `human_subject`, `human_obligation`, `human_condition`, `human_threshold`, `human_deadline`, `human_exception`.
4. There is also a `notes` column — use it for anything you find ambiguous.
5. **If some of the CSV rows also have `claude_*` columns visible, HIDE those columns before you start.** In Excel: select columns with `claude_` prefix → right-click → Hide. Looking at them first biases your judgment.
6. When you finish, save the file as **CSV (UTF-8)** and send it back.

---

## The six fields — strict definitions

Read these once carefully. Then keep them open while you annotate.

### 1. `human_subject` — Who is regulated?
The actor that must do something. Usually a noun phrase: `bank`, `broker-dealer`, `controller`, `firm`, `investment adviser`, `supervisor`, `data processor`, `Member State`.

**Rule:** use the shortest noun phrase from the text that names the regulated party. Lowercase. Drop articles ("a", "the").

| Source text says | You write |
|---|---|
| "A bank shall calculate..." | `bank` |
| "The controller must notify..." | `controller` |
| "Broker-dealers are required to..." | `broker-dealer` |
| "Each Member State shall ensure..." | `member state` |

**Leave blank** if no specific actor is named (e.g., text describes what "the regulation" does without pointing to any entity).

### 2. `human_obligation` — What action is required?
The core verb phrase that states what must be done. Start with a verb. Keep it short — one clause, not the whole sentence.

**Rule:** include the verb and its direct object. Drop adverbials and subordinate clauses (those go into `condition` or `deadline`).

| Source text says | You write |
|---|---|
| "shall calculate credit equivalent amounts" | `calculate credit equivalent amounts` |
| "must disclose all material conflicts of interest" | `disclose all material conflicts of interest` |
| "shall notify the supervisory authority of a breach" | `notify the supervisory authority of a breach` |
| "are prohibited from accepting commissions" | `not accept commissions` (render prohibitions as "not X") |

**Leave blank** if the text describes rather than requires — e.g., "The framework was established in 1988."

### 3. `human_condition` — When/where does the obligation trigger?
The scope qualifier — who, what, or under what circumstances the obligation applies. Usually starts in the source with "for", "where", "when", "if", "in respect of", "to".

**Rule:** lift the prepositional/subordinate clause that narrows the obligation.

| Source text says | You write |
|---|---|
| "for OTC derivatives..." | `for OTC derivatives` |
| "where personal data is processed" | `where personal data is processed` |
| "when an adverse event occurs" | `when an adverse event occurs` |
| "to retail customers" | `to retail customers` |

**Leave blank** if the obligation applies unconditionally.

### 4. `human_threshold` — Any numeric or quantitative limit?
Monetary amounts, percentages, counts, sizes, ratios. Anything measurable.

**Rule:** lift the number + its unit.

| Source text says | You write |
|---|---|
| "at least 10% of capital" | `at least 10%` |
| "USD 5 million" | `USD 5 million` |
| "150 basis points" | `150 basis points` |
| "minimum two business days" | `minimum two business days` |

**Leave blank** if no numeric/quantitative element is present.

**Tricky:** a pure date (e.g., "January 2025") is a **deadline**, not a threshold. A duration limit (e.g., "within 72 hours") is a **deadline**. A quantity (e.g., "more than 100 employees") is a **threshold**.

### 5. `human_deadline` — Any time constraint?
When the obligation must be met. Durations ("within 72 hours"), dates ("by 1 January 2025"), ordinal triggers ("prior to engagement", "before reporting period").

**Rule:** lift the time-limit phrase.

| Source text says | You write |
|---|---|
| "within 72 hours" | `within 72 hours` |
| "by 1 January 2025" | `by 1 January 2025` |
| "prior to engagement" | `prior to engagement` |
| "before the reporting period" | `before the reporting period` |
| "at least annually" | `at least annually` |

**Leave blank** if no time element is present.

### 6. `human_exception` — Any carve-out or exemption?
The "unless/except" clause, or anything that releases the subject from the obligation under specific conditions.

**Rule:** lift the exemption phrase (usually after "unless", "except", "save that", "other than", "provided that" when it weakens rather than triggers).

| Source text says | You write |
|---|---|
| "unless authorised by the supervisor" | `unless authorised by the supervisor` |
| "except for small institutions" | `except for small institutions` |
| "save where the data is anonymised" | `save where the data is anonymised` |

**Leave blank** if no exception is present. This will be the most commonly blank field.

---

## Decision rules to keep you consistent

1. **Use words from the source text.** Do not paraphrase. "Calculate" stays `calculate`, not `compute`.

2. **Prefer shorter answers.** `bank` beats `a bank operating in the EEA`. Shorter phrases reduce disagreement.

3. **Don't infer.** If the field is not literally in the clause, leave it blank. This is the #1 rule that improves kappa — annotators disagree most on things that aren't in the text but "might be implied."

4. **One obligation per row.** If a clause contains two requirements, fill in the primary (structurally main) one and note the other in the `notes` column.

5. **When text is descriptive, not prescriptive:** leave `obligation` blank. Texts like "The framework was established..." or "Market participants have noted..." are not obligations.

6. **For prohibitions:** render as "not X" in the obligation field. "Broker-dealers are prohibited from receiving commissions" → `not receive commissions`.

7. **Use `notes` liberally** for anything you're unsure about. Examples of good notes:
   - "ambiguous subject — could be bank or supervisor"
   - "has two co-equal obligations; annotated the first"
   - "references section 4 which may contain additional conditions"

---

## Three fully worked examples

### Example A (Basel III)

**source_text:** "A bank shall calculate credit equivalent amounts of OTC derivatives, exchange traded derivatives and long-settlement transactions that expose a bank to counterparty credit risk under the counterparty credit risk standards."

| Field | Answer |
|---|---|
| `human_subject` | `bank` |
| `human_obligation` | `calculate credit equivalent amounts` |
| `human_condition` | `for OTC derivatives, exchange traded derivatives and long-settlement transactions that expose a bank to counterparty credit risk` |
| `human_threshold` | *(blank)* |
| `human_deadline` | *(blank)* |
| `human_exception` | *(blank)* |

### Example B (Reg BI)

**source_text:** "A broker-dealer must disclose all material conflicts of interest to retail customers prior to or at the time of recommendation, unless the conflict has been eliminated."

| Field | Answer |
|---|---|
| `human_subject` | `broker-dealer` |
| `human_obligation` | `disclose all material conflicts of interest` |
| `human_condition` | `to retail customers` |
| `human_threshold` | *(blank)* |
| `human_deadline` | `prior to or at the time of recommendation` |
| `human_exception` | `unless the conflict has been eliminated` |

### Example C (GDPR)

**source_text:** "The controller shall notify the supervisory authority of a personal data breach without undue delay and, where feasible, not later than 72 hours after having become aware of it."

| Field | Answer |
|---|---|
| `human_subject` | `controller` |
| `human_obligation` | `notify the supervisory authority of a personal data breach` |
| `human_condition` | *(blank — no scope qualifier)* |
| `human_threshold` | *(blank)* |
| `human_deadline` | `without undue delay and, where feasible, not later than 72 hours after having become aware of it` |
| `human_exception` | *(blank)* |

---

## Time expectation

At ~45 seconds per row:
- 300 rows ≈ 3.5–4 hours
- 100 rows ≈ 75 minutes
- 50 rows ≈ 40 minutes

Take breaks every 50 rows. Tired annotators produce worse data than slow annotators.

---

## When you finish

Save the file as CSV (UTF-8 encoding) and send it back. The filename should stay the same (`kappa_300_worksheet.csv`, `regbi_gold_worksheet.csv`, or `gdpr_human_worksheet.csv`).

Questions? Ask before you start, not halfway through.

# Classifier prompt decision log (maintainer document, NOT a prompt)
This file holds the decision tracking that previously sat in HTML comments at the top of the two prompt files. The prompt files `main_intent_classifier.md` and `lead_qualification_prompt.md` are the only authoritative prompts and now contain only classifier instructions. Nothing here is sent to a model. Production parity is not established; no accuracy claim is made for either prompt.

## Main Intent Classifier (`main_intent_classifier.md`)

```text
STATUS: business decisions D1 and R1-R5 CONFIRMED by the project owner. Only non-blocking point O1 remains open. Not evaluated; no accuracy claim is made.

CONFIRMED BUSINESS DECISION
D1. Follow-up on a prior handoff with no new explicit request ("any update?", "did anyone call me?") is QUERY.
    The original text contradicted itself: the Step 2 note, Step 6 and the QUERY definition said QUERY, while one example said HUMAN_HANDOFF_REQUEST.
    Confirmed: QUERY, because asking for an update is not itself a request to be connected to a human.
    Benchmark note: the earlier plan to exclude such cases from scored metrics pending this decision no longer needs to apply to the rule itself; any change to dataset exclusions must be made separately.

CONTRADICTIONS RESOLVED FROM THE ORIGINAL TEXT (confirmed by the project owner; each follows the majority of the original explicit rules)
R1. Accepting a counsellor-connect offer: the original definition and the closing note said a reply to an offer "is NEVER by itself enough", but Step 2.2, Step 4, the category definition's opening sentence and two examples all said a clear affirmative to a specific counsellor-connect offer IS a handoff. The draft keeps the latter and removes the contradicting sentences.
R2. Declining an offer: the GREETING definition listed "no thanks / not now" as GREETING, while Step 4 sent "any type of reply" to an offer to QUERY. The draft treats an explicit decline with no new question as GREETING, and every other reply to a question or offer as QUERY.
R3. Fallback: the original Step 7 said GREETING while the QUERY definition said "when in doubt -> QUERY". The draft uses QUERY.
R4. Step 2.2 and Step 4 duplicated the counsellor-offer rule. It now lives only in Step 2.
R5. "hello"/"hi" were GREETING in an example but absent from the Step 5 token list. They are now in the list.

NON-BLOCKING OPEN POINTS
O1. A bare "no" or "thanks" replying to a non-counsellor offer is QUERY by Step 4 (unchanged original wording). Only explicit decline phrases are GREETING. Confirm if you want a bare "no" to count as a decline.
O2. Step 3 treats "application status" questions as QUERY, not UPLOAD_STATUS_CHECK (unchanged original behaviour, now stated explicitly).

INPUT FORMAT
The Input section describes the logical inputs only. It does not claim to match the production Dify serialization, which is unverified (production parity not established).

EXAMPLES
All examples below are hand-written SYNTHETIC examples, not real student messages.
```

## Lead Qualifier (`lead_qualification_prompt.md`)

```text
STATUS: DRAFT — NOT FINALIZED. L1, L2, L3 and L9 are CONFIRMED by the project owner. L4-L8 remain open. Not evaluated; no accuracy claim is made.

SCOPE OF THIS EDIT
Tier A and Tier B membership is unchanged from the original prompt. Edits only clarify wording: the threshold logic, acknowledgement handling, the uncertainty fallback, output and failure handling, and the input description.

OPEN BUSINESS DECISIONS (each can change whether need_cta is true)
L1. [CONFIRMED] Repeated wording of the same enquiry is not a second distinct signal (asking one programme's fee twice counts once). The prompt (rule 3 and an example) now says so. Builder follow-up (no code changed yet): it already counts such a repeat once, but still emits the ambiguity reason REPEAT_COUNTING_RULE_UNAPPROVED, which is now stale. It also emits only an ambiguity, with no action, when the same programme's entry requirements are asked twice with no fee question.
L2. [CONFIRMED] Keep the existing Tier A logistics signals (hostel, transport, campus location, part-time, working while studying, visits) and flag their broad scope. Prompt unchanged; the builder already flags A16 with BROAD_SCOPE_REVIEW.
L3. [CONFIRMED] Specific results, grades, scores or CGPA = Tier A; a qualification or academic status without specific results = Tier B. Already written as the Overlap rule in the prompt.
L4. Objection signals (price pushback, recognition questions, "let me think about it" after naming a programme) require judgement. The prompt can only apply them if the summary states them explicitly.
L5. "Returned to the chat in a later session" counts only if the summary says so. Whether production supplies this is unknown.
L6. Pathway signals need a qualification AND a "what can I do with it" question. If the summary lists them as separate items, this draft treats them together. Confirm.
L7. "Asked about the deadline AFTER naming a specific programme" needs order information that a summary may not carry.
L8. "Offered to submit or upload" documents: what counts as an offer is not defined in the original.
L9. [CONFIRMED] A clearly stated Tier A action is true; genuinely unclear signals do not count. Already written as decision rule 5 in the prompt.
Note: "credit transfer" appears twice in Tier A (transfer signals and ownership signals). Both are Tier A, so the duplicate has no effect on the result.

INPUT FORMAT
The Input section describes the logical input only. It does not claim to match the production summary builder or the production Dify flow, both unverified (production parity not established). This prompt does not guarantee that student actions were extracted correctly into the summary.

EXAMPLES
All examples are hand-written SYNTHETIC examples, not real student data.
```

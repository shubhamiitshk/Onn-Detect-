# ONN startup strategy and first-customer plan

Prepared 31 October 2026. Decision brief for a small-budget launch, early pitch, and eventual growth.

This is a proposed commercial direction, not a claim that the proposed product is implemented or that customer demand has been validated. No customers were contacted. Prices, targets, and budgets below are experiments. Working product names have not been checked for availability.

## 1. Recommendation

Build a narrow evidence-to-report workflow for one type of business. My default first experiment is **ONN Handover: documented vehicle checkout and return for small rental operators**.

The promise: **Help staff capture the required condition records, find what is missing, compare checkout and return, and prepare a report the owner can review.**

This gives the company something useful to sell while detector accuracy is independently evaluated. A business can measure report time, missing photos, repeat work, and retrieval time without trusting an unvalidated fake/real verdict.

Choose this segment only as a starting hypothesis. The selection gate is access to willing buyers with an observed recurring problem, not our preference or the amount of code that can be reused. If the founder can obtain substantially better access to independent inspection agencies or e-commerce sellers, test that access before committing. Pick one segment for the first pilot cycle; do not build several products at once.

ONN's current assets are useful: Android media import and sharing, bounded local processing, native model execution, scan reports, source-linked chat, and uncertainty-aware output. They are a technical foundation. Product adoption and commercial differentiation still need to be earned.

## 2. What research changes about the opportunity

Generic deepfake detection is an established competitive category. Reality Defender offers submitted-media analysis and products for calls and meetings; Hive offers synthetic-media APIs and free detection tools. This is evidence of competing offerings, not independent evidence of their accuracy or ONN's addressable market. See [Reality Defender's platform](https://www.realitydefender.com/platform/technology) and [Hive's product](https://hivemoderation.com/ai-generated-content-detection).

Inspection evidence is also competitive. [Truepic](https://www.truepic.com/vision/insurance) combines authenticated capture and AI-extracted inspection findings for insurance. [Record360](https://record360.com/lp/auto-rental/) explicitly offers vehicle checkout/return checklists, photo/video records, comparisons, and inspection handoff. [CompanyCam](https://companycam.com/pricing) lists photo reports, project histories, offline work, and AI document features. [Fleetio](https://help.fleetio.com/en_US/fleetio-go-for-account-owners-admins/use-fleetio-go-offline) documents offline inspections.

Consequently, offline operation, a chatbot, photos, AI summaries, and PDF reports alone do not establish a unique product. The opening to test is a customer group whose current workflow remains inconvenient despite these alternatives. Possible reasons include setup effort, an existing client's report format, local support, particular Android devices, language needs, or a costly handoff. None of those gaps is confirmed yet.

Ask prospects to try an incumbent or explain why they rejected it. If an existing tool solves the problem well at an acceptable price, ONN needs a different wedge. Low price by itself is vulnerable to support costs and imitation.

## 3. Three commercial directions

| Candidate | First customer to investigate | Specific job | Why it could fit | Main uncertainty |
|---|---|---|---|---|
| **ONN Handover — default experiment** | Owner of a small vehicle rental operator with frequent checkout/return activity | Produce a complete, comparable condition record and retrieve it during a query | Frequent mobile workflow; measurable completeness and preparation time | Strong incumbents; staff may resist extra capture steps |
| **ONN Evidence Desk — closest reuse** | Independent inspection or investigation agency preparing recurring client reports | Organize a case's files and notes into a source-linked draft for human approval | Existing imports, reports, and chat are relevant | Need durable originals, better document handling, and an output clients accept |
| **ONN Return Desk — a different idea** | E-commerce seller or return-processing agency with repeated disputed returns | Assemble order details, return photos, identifiers, and selected unboxing-video timestamps into a review packet | Direct connection to costly operational work; seller communities can provide distribution | Marketplace rules, current competition, actual case volume, and seller willingness to pay |

For Return Desk, begin with one seller and one marketplace's current workflow. Organize the seller's own evidence and prepare a human-reviewed submission; do not promise reimbursement or automatic claim acceptance. Export originals, clearly identified derived copies, relevant timestamps, and an editable factual description in the forms actually required by that programme. A PDF alone may not meet submission requirements. Capture dispatch/packing records as well as return-opening records: software cannot recreate missing prior evidence. Preserve continuous original footage and do not generate or retouch supposed damage evidence.

Amazon India's [historical Easy Ship SAFE-T explainer](https://sell.amazon.in/ml/seller-blog/here-is-how-easy-ship-sellers-can-file-safe-t-claims) illustrates that a seller-claim workflow exists, but its 2018 instructions are not current policy. Current eligibility, required evidence, deadlines, permissions, and submission rules would need verification before implementation. [Unilytic's own offering](https://unilytic.in/policies/amazon-safe-t-claim-guide/) also shows relevant existing competition in Indian returns and claims workflows; its descriptions are not authoritative marketplace policy.

An additional, more distant pivot is an offline service-manual assistant for a particular repair trade: technicians ask a question, attach a machine photo, and retrieve the exact approved manual passage. It needs customer-authorized manuals, reliable retrieval, and task-specific evaluation. It is less connected to ONN's detection work and should be considered only if those buyers are easier to reach.

A consumer family-scam app could remain a later distribution experiment. Its payment willingness, occasional usage, acquisition cost, and risk of false reassurance need validation. A screenshot cannot establish payment receipt, and a low spoof score cannot establish who is speaking. Do not make these the first revenue dependency.

## 4. Recommended first customer and workflow

Proposed initial customer profile:

- One local vehicle rental operator with roughly 10–100 vehicles, frequent handovers, and an owner who can authorize a trial. This range is a prospecting filter, not market research.
- Staff currently use some combination of phone galleries, chat messages, paper forms, or spreadsheets and can demonstrate a specific difficulty.
- Willing to show consented, appropriately redacted examples and let the founder observe checkout and return.
- Enough eligible handovers to evaluate repeated use within four weeks.

User: the counter or yard staff member. Buyer: the owner or operations manager. Reviewer: the person handling a customer query, maintenance request, or disputed condition report.

For this plan, a **record** means one completed inspection event: checkout or return. A matched checkout/return pair is two records. Record both event time and report-preparation time; do not count a pair as one record in one metric and two in another.

The proposed product flow is:

1. Start a checkout record linked to vehicle and rental reference.
2. Follow the operator's required photo checklist. Record observations as explicit staff statements.
3. Show missing required views before completion. Add optional annotations and customer acknowledgement.
4. On return, show matching checkout views beside new views. A person records observed differences.
5. Prepare a branded, reviewable report with source references and export it through the operator's existing process.
6. Retrieve the record later and answer bounded questions such as, “Which checkout image shows the rear bumper?”

A memorable first feature is **“What evidence are we still missing?”** It can begin as a dependable checklist. Visual AI may suggest blur, framing, or a possible mismatch after evaluation; it should not silently certify the vehicle's condition.

Avoid automated damage attribution, repair valuation, fraud accusations, or assertions that an acknowledgement proves responsibility. Basic documentation does not establish those conclusions.

## 5. MVP scope and implementation boundaries

Build after initial discovery, and only for the chosen workflow:

| Capability | Pilot requirement | AI dependence |
|---|---|---|
| Case/asset identity | Searchable vehicle/rental record; explicit checkout/return association | None |
| Required evidence | One operator-approved template; explicit missing items and completion override reason | None |
| Media preservation | Case-owned originals, bounded previews, storage controls, recovery/export, and deletion | None |
| Comparison | Human-reviewed side-by-side views with clear labels | None for first version |
| Report | Client-approved layout; source item IDs; draft, corrections, and approved version | Optional drafting |
| Case questions | Answers link to supporting evidence; “not recorded” when absent | Optional local model |
| Handoff | Usable report and evidence export in the buyer's existing process | None |

The current application is not already this product. Existing history stores assessments rather than a complete durable case vault. Guided checkout/return capture, asset-linked originals, report templates, and review history need implementation.

Current multimedia limits also matter: audio attachments do not provide transcription; PDF text handling is bounded and platform-dependent; video analysis samples frames; the small local generator is not a validated expert. The latest [chat implementation verification](CHAT_IMPORT_UI_FIX_2026-10-30.md) does not establish physical-phone detector accuracy. Do not pitch comprehensive video search, reliable serial-number extraction, or unlimited document understanding as shipped features.

For a first report, typed structured observations and deterministic templates can deliver value before voice transcription or model-generated prose. Keep ordinary recording and export usable if the optional model is absent or slow.

A proposed architecture separates case originals and immutable assessment snapshots from editable notes, AI suggestions, and approved reports. Human corrections create new report versions. Citations identify actual source items. User-entered dates and locations remain distinguishable from measured metadata and independently verified claims. An exported package must work on a second recipient device, with references resolving to bundled evidence rather than app-private paths.

A file digest identifies bytes; it does not prove a truthful scene, trusted capture time, or honest actor. C2PA likewise distinguishes provenance verification from factual truth; see its [official explainer](https://spec.c2pa.org/specifications/specifications/2.3/explainer/Explainer.html).

The repository currently excludes network permissions. Discover whether checkout and return happen on one shared company phone or different staff devices. A pilot needs either an agreed shared-device process or a working, explicit handoff that preserves the matching records and originals. Begin with user-controlled export/import where that fits the observed workflow. If handoff is a repeated blocker, a shared workspace becomes core scope rather than an optional decoration. Design any network-enabled product deliberately with clear transfer controls; do not describe it as zero-network.

## 6. First customers on a small budget

Start in one city or a reachable professional network. Build a list of 30 named businesses with an owner contact, estimated handover frequency, present tools, and a reason to investigate. Public listings, local introductions, relevant business associations, and permitted community participation are prospecting channels; a list alone is not traction.

Target 15 substantive conversations and observe at least five actual workflows. Ask:

- Show me the last handover record you had difficulty using. What happened?
- How many such records do you create each week? Who reviews them?
- What is the slowest step? How many minutes does it take?
- What evidence was missing in the last three disputed or incomplete cases?
- Which existing tools have you tried? Why are you using the current process?
- Who can approve a paid trial? What output and price would justify renewal?

Collect consented/redacted examples and baseline observations, not just compliments about the idea. Find the economic buyer rather than relying only on an enthusiastic user.

Offer a bounded four-week assisted pilot: one location, one report template, an agreed case cap, setup support, and a final scorecard. During an early concierge trial, the founder can help prepare reports manually; disclose that assistance and record its time. Do not represent manual work as shipped automation. Convert the repeated steps into software.

The founder should conduct outreach personally and with the prospect's consent. This brief does not send messages or enroll anyone.

## 7. Pricing and economics to test

Use a **₹5,000 four-week pilot for one location and up to 30 records** as an opening experiment, including setup and a template. Adjust based on scope and interviews. This is a proposed offer, not an established market price. Test willingness to renew around ₹2,000–₹5,000 per location per month only after measuring value and support costs. Actual case volume may require a different unit or cap. A price that exceeds the measured value is a reason to change the offer or segment.

Example value calculation, entirely hypothetical:

`300 records/month × 5 active minutes saved ÷ 60 × ₹200/hour = ₹5,000/month of recovered staff capacity`

This is recovered capacity, not necessarily reduced payroll or cash savings. It would not justify a ₹5,000 monthly fee on time savings alone; a lower price or additional measured value would be needed. Replace all three inputs with the operator's real baseline. Count avoided travel or prevented disputes only when actually observed and attributable. Measure the entire workflow: extra capture, corrections, sharing, and finding old records can offset report-time savings. Compare switching value against suitable incumbent products as well as galleries and paper.

Track media storage, support time, onboarding, maintenance, payment collection, and any future synchronization costs. Local inference avoids a hosted per-token generation bill for that operation; it does not make the company cost-free.

Set a first-month cash cap the founder can afford. An illustrative cap is ₹10,000, excluding existing equipment and founder time: ₹4,000 for visits/calls, ₹2,000 for basic demo materials or web presence, ₹2,000 for pilot incidentals, and ₹2,000 contingency. These are allocation limits, not supplier quotes. Use existing phones before buying a device fleet. Avoid paid acquisition until a repeatable pilot-to-renewal path emerges.

## 8. Ninety-day plan with decision gates

| Period | Work | Evidence needed to move forward |
|---|---|---|
| Days 1–14 | Interview 15 buyers/users in the selected lane; observe five workflows; build one example report from authorized samples | Three operators describe the same recurring problem and agree to a concrete pilot discussion; at least one accepts a paid pilot or deposit |
| Days 15–30 | Deliver an assisted checklist-and-report pilot using existing capture and manual comparison; establish reliable preservation and handoff before using live customer originals | Actual repeated use, baseline and pilot timings, review corrections, and accepted output |
| Days 31–60 | Run three paid pilots, aiming for at least 50 eligible records across them; reduce founder assistance | Net time savings, no lost original evidence, acceptable correction burden, and use during normal operations |
| Days 61–90 | Seek renewals and referrals; improve the largest measured bottleneck | At least two of three pilot buyers renew at a stated price and use the product for most eligible work |

These are proposed small-sample decision rules, not statistical proof or promised results. Define eligible records and measures before starting. A useful initial target is at least 30% lower median active documentation time without worse completeness or an unacceptable slowdown at the counter. Report sample size, range, and exceptions; do not turn one successful demonstration into an accuracy claim.

The first month is not a promise to deliver a new media vault, rich chat, automatic comparison, and multi-user product simultaneously. Use a disclosed assisted process for learning, then implement the most repeated bottleneck. Case chat can wait until customers repeatedly use the core checklist and report.

Change direction if the founder cannot obtain buyer access, if repeated interviews do not expose the same costly problem, if incumbents already solve it satisfactorily, or if staff stop using the pilot. When a specific objection repeats, test a revised offer once. More AI features do not resolve a missing buyer or absent workflow value.

## 9. Pitch for the first launch

Proposed short pitch:

> ONN helps small vehicle rental teams turn checkout and return photos into complete, reviewable condition records. Staff follow a guided checklist, compare matching views, and prepare a report with links to the original evidence. Our starting advantage is a working private Android foundation; our next milestone is demonstrating less documentation work with three paid pilot locations. We are seeking pilot partners and introductions to operators who handle frequent vehicle handovers.

Use a seven-slide pitch:

1. **Problem:** one real, consented customer workflow and its measured difficulty.
2. **Customer:** the exact operator profile, staff user, buyer, and purchasing trigger.
3. **Product:** a short checkout-to-return-to-report demonstration.
4. **Alternatives:** current chat/gallery/paper process and actual competitors; explain the tested switching reason.
5. **Business model:** pilot offer, renewal hypothesis, delivery costs, and measured customer value when available.
6. **Evidence:** functioning prototype today; paid use, retention, and time savings only when actually obtained. Separate plans from results.
7. **Ask and milestones:** introductions, specific pilot commitments, and a costed amount of support if seeking funding.

The existing [historical pitch](PITCH_DECK_Z21_KGP_FORGE.md) contains claims about readiness, universal latency, SDK availability, compliance, hardware assurance, and market size that require fresh substantiation. Use this brief as a new positioning draft; do not copy unsupported claims into a new application. Tests show defined software behavior, not market traction, forensic validity, regulatory compliance, or actual revenue.

Build an initial market estimate from a verified reachable list and tested pricing. For example, `100 named eligible locations × ₹5,000/month = ₹60 lakh annual revenue at full conversion` is arithmetic for a hypothetical account pool, not TAM, a forecast, or expected revenue. Model realistic conversion and churn separately after pilots.

## 10. Growth after evidence of repeat use

The initial advantage would be fast setup around a specific accepted report and a process staff actually complete. A durable business could develop through permissioned workflow data, useful templates, reliable integrations, customer referrals, and proven outcomes. Sensitive customer evidence is not an asset the company can automatically reuse or train on; obtain explicit authorization and rights.

Expand in order: more locations at the same buyer, referrals to similar operators, an appropriate channel partner, then a closely related workflow. Add a reviewer dashboard or integrations when repeated customer work justifies them. Product names, user counts, model swaps, and download counts are not substitutes for renewals.

Detection can become a separately evaluated capability later. Measure exact supported media and target phones, meaningful false-positive rates, coverage/abstentions, and transformation robustness. Resolve the relevant trust and release findings before making stronger assurance claims. Keep a person responsible for consequential interpretations.

The immediate founder deliverable is one paid, repeated, useful workflow with a before/after scorecard. That is the evidence that should shape the next build and the next pitch.

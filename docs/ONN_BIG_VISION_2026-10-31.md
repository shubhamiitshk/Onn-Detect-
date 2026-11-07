# ONN: the larger company thesis

31 October 2026. Strategic proposal, not a claim of implemented capability or validated demand. This complements the [first-customer plan](STARTUP_STRATEGY_2026-10-31.md); it raises the long-term ambition while retaining a narrow commercial experiment.

## The company to explore

**ONN could build evidence verification infrastructure for remote business decisions.**

Companies receive photographs, videos, documents, and recordings to support claims about the physical world. ONN would help a reviewer answer: what supports this case, which facts remain unestablished, and which additional evidence would be useful before acting?

The proposed outcome is a reviewable evidence record, clear policy checks, and a next action. A synthetic-media detector is one fallible measurement in that process. The local chat interface helps a person inspect the record; it does not establish facts by generating an answer.

The ambition can extend from a single case-review product to APIs and capture components embedded in other businesses' software. Software agents could eventually consume the same structured records, subject to the customer's approval rules. This is an expansion hypothesis, not evidence of an existing AI-agent market for ONN.

## The distinctive product question

**What should we ask for next to resolve the uncertainty?**

Illustrative warranty workflow, entirely proposed:

1. A service team receives a damage photograph and invoice.
2. ONN records original files, declared claims, and exact source references.
3. It compares reviewer-confirmed identifiers with an authorized order or service record.
4. It reports that the submitted evidence does not establish a required condition, such as a matching serial number or an adequate view of damage.
5. A reviewer requests a specific new photograph, document, or guided capture session.
6. The new evidence joins the case history, and the reviewer records the decision and rationale.

A guided capture challenge can add information; it cannot guarantee an honest scene or defeat every generative attack. Device signatures bind particular bytes and assertions under their verified trust model, rather than proving that a claim is true. Supported checks and limitations must remain explicit.

## The product layers

| Layer | Purpose | Important boundary |
|---|---|---|
| Capture and intake | Accept existing evidence; optionally request guided new material | Distinguish imported media from a supported capture session |
| Evidence analysis | Apply supported provenance, quality, duplication, and detector checks | A missing check or uncertain result stays unavailable or uncertain |
| Business context | Relate evidence to an authorized order, asset, service job, or case | A copied invoice and a verified business record have different authority |
| Review and requests | Apply versioned requirements; propose additional evidence; record human approval | Do not automatically reject a claim based on an unvalidated detector |
| Integration | Return structured evidence status and case history to customer software | An exported record must preserve source identity and actual verification scope |

Avoid a universal truth score. “The invoice identifier matches the authorized order,” “this file lacks supported provenance,” and “the device capture session was not verified” are distinct facts with different meanings.

## Why this could become substantial

A case-review product can lead to an embedded infrastructure business if customers repeatedly need the same evidence checks and integrate them into their existing systems. Adoption would follow a single workflow, additional teams or locations, an integration partner, and then a closely related industry.

An API and SDK are distribution forms, not an automatic competitive advantage. They matter when another product has a concrete reason to embed ONN rather than build the capability or choose a competing vendor.

The potential defensibility is accumulated expertise around a workflow: permissioned examples with independently established outcomes, measured failure cases, useful evidence-request policies, accepted integrations, and dependable operation on relevant devices. Customer corrections are not automatically ground truth. Sensitive evidence is not automatically available for training or cross-customer search. Rights and authorization must be established.

Privacy-preserving local processing could be valuable where a buyer actually needs it. Current Android operation provides a foundation, but a shared review service requires deliberate additional infrastructure and transfer controls. A connected product must not be advertised as zero-network.

## The competition is real

[Truepic's insurance product](https://www.truepic.com/vision/insurance) already combines inspection requests, authenticated capture, AI findings, and integrations. [Reality Defender](https://www.realitydefender.com/platform/technology) offers media findings in calls, meetings, APIs, and analyst workflows. These primary sources establish offered products, not independent performance comparisons.

Therefore, “trust platform,” “AI plus provenance,” “request better photos,” or “API plus mobile SDK” does not demonstrate uniqueness. ONN needs a buyer-visible reason to switch. Test private processing requirements, a specific unmet case workflow, integration effort, evidence-request efficiency, accepted reporting formats, and local deployment economics. These are candidate differentiators requiring direct comparison, not assertions that competitors lack those features.

The standard itself also limits the claim: [C2PA explains](https://spec.c2pa.org/specifications/specifications/2.3/explainer/Explainer.html) that provenance does not establish whether depicted content is factually true. Product language should preserve that distinction.

## The first commercial experiment

For the infrastructure thesis, a useful first segment to investigate is **one type of warranty or inspection evidence review**, sold through a service/inspection agency already doing the work. Independent claims-processing or survey teams are another route when the founder can reach their actual buyer.

Select between those channels based on observed access, repeat case volume, purchasing authority, and the ability to obtain authorized examples. No existing buyer relationships or sales-cycle assumptions have been established here. Starting directly with a large insurer may require more procurement and assurance work than the initial budget supports; test the purchasing process before assuming an agency route is easy.

The first paid promise is less preparation/review work and more complete evidence. Keep adverse financial decisions with the authorized reviewer. Measure the full cost of requesting additional evidence, including claimant effort and completion/drop-off rates, rather than counting fewer internal minutes alone.

A demonstration can use one case type, one evidence checklist, one permitted reference record, source-linked notes, and a reviewed additional-evidence request. Manual assistance is acceptable when disclosed and measured. It does not require a general autonomous agent or a newly trained foundation model.

## Execution gates

1. **Demand:** 15 substantive conversations, observed recurring difficulty, and a buyer accepting a bounded paid trial. Compare against the incumbent process and relevant commercial alternatives.
2. **Utility:** three paid pilots in the same workflow; total preparation/review time, completeness, corrections, additional requests, and requester burden measured before and during the pilot.
3. **Retention:** at least two buyers renew at a stated price and repeatedly use ONN for eligible work. These small-sample rules guide the next investment; they are not proof of market scale.
4. **Reliability:** preservation and recovery of originals, source-linked exports, access/deletion controls, usable staff handoff, and explicit failure handling.
5. **Assurance:** independently evaluated supported detector routes, target-device measurements, relevant provenance/security fixes, and customer-specific acceptance criteria before selling stronger authenticity or fraud-detection claims.
6. **Infrastructure:** one real integration driven by repeated customer use; build a versioned API/SDK only around the contracts that proved useful.

Pricing initially follows a bounded paid case-review pilot. Later test an annual platform commitment plus case usage or private-deployment support when the delivery costs and purchasing process justify it. These are business-model hypotheses. Avoid presenting an unsupported revenue forecast as market evidence.

## Current foundation and missing work

ONN has a working Android prototype with local media processing and model chat. Its latest [verification record](CHAT_IMPORT_UI_FIX_2026-10-30.md) does not establish independent physical-phone forensic accuracy. It is not yet the described infrastructure platform.

The necessary new capabilities include durable case-owned originals, verified source snapshots, review state and evidence requests, reliable recipient-side exports, authorized business-record access, team handoff, and eventually a properly packaged integration surface. Stronger authenticity claims require the relevant trust and release issues to be resolved and independently tested.

The previous vehicle-handover proposal remains a possible inexpensive workflow experiment. It should become the starting point only if buyer access and observed demand are stronger there. It does not limit the long-term company to rental reporting. Conversely, calling the company infrastructure does not remove the need to win one concrete customer workflow.

## Proposed pitch

> ONN is building verification infrastructure for businesses that make decisions from photos, videos, and documents. We help reviewers connect evidence to a case, identify what remains unestablished, and request the next useful check. We are starting with one repeat evidence-review workflow, using a private Android prototype as our foundation. Our next milestone is paid pilot use with measured review time, evidence completeness, and renewals; our longer-term product is an integration layer other business software can embed.

The funding story should tie requested support to these gates, actual costs, and a credible team plan. The big vision is the destination; observed customer behavior should decide the next build.

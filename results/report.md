# Evaluation report

Mode: MOCK (offline heuristic)  |  Incoming emails: 20  |  Retrieved examples per email: 3

## Overall scores (0-100)

| System | Overall | Guardrail pass |
|---|---|---|
| Reference (human ceiling) | 87.5 | 100.0% |
| Model (retrieval-grounded) | 8.8 | 100.0% |
| Floor (degenerate) | 0.0 | 0.0% |

A valid metric must rank reference >= model >> floor. Here: 87.5 >= 8.8 >> 0.0.

## Model per-axis average (1-5)

| Axis | Avg |
|---|---|
| resolution | 1.35 |
| grounding | 1.35 |
| completeness | 1.35 |
| tone | 1.35 |
| clarity | 1.35 |

## Model score by category

| Category | Score |
|---|---|
| hr_recruiting | 8.3 |
| request_info | 8.3 |
| other | 0.0 |
| negotiation_deal | 8.3 |
| scheduling | 0.0 |
| approval_review | 12.5 |
| personal_social | 25.0 |
| status_update | 12.5 |

## Per-response (model)

| Incoming | Retrieved from | Score | Guardrails |
|---|---|---|---|
| in-0001 | hist-0007, hist-0134, hist-0167 | 0.0 | pass |
| in-0002 | hist-0170, hist-0045, hist-0200 | 0.0 | pass |
| in-0003 | hist-0166, hist-0053, hist-0093 | 0.0 | pass |
| in-0004 | hist-0180, hist-0054, hist-0153 | 25.0 | pass |
| in-0005 | hist-0187, hist-0200, hist-0173 | 0.0 | pass |
| in-0006 | hist-0026, hist-0045, hist-0143 | 25.0 | pass |
| in-0007 | hist-0135, hist-0067, hist-0059 | 25.0 | pass |
| in-0008 | hist-0156, hist-0059, hist-0099 | 0.0 | pass |
| in-0009 | hist-0059, hist-0006, hist-0128 | 25.0 | pass |
| in-0010 | hist-0011, hist-0045, hist-0110 | 25.0 | pass |
| in-0011 | hist-0159, hist-0156, hist-0078 | 0.0 | pass |
| in-0012 | hist-0072, hist-0109, hist-0016 | 0.0 | pass |
| in-0013 | hist-0066, hist-0081, hist-0193 | 0.0 | pass |
| in-0014 | hist-0179, hist-0060, hist-0034 | 0.0 | pass |
| in-0015 | hist-0014, hist-0120, hist-0045 | 25.0 | pass |
| in-0016 | hist-0199, hist-0087, hist-0045 | 25.0 | pass |
| in-0017 | hist-0066, hist-0198, hist-0081 | 0.0 | pass |
| in-0018 | hist-0081, hist-0135, hist-0066 | 0.0 | pass |
| in-0019 | hist-0152, hist-0028, hist-0006 | 0.0 | pass |
| in-0020 | hist-0162, hist-0045, hist-0170 | 0.0 | pass |
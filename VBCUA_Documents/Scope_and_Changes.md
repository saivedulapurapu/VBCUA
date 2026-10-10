# Scope and changes since June 2026

The PDFs in this folder are the original project record from June 2026 and are kept unchanged.
This note lists where they differ from the current code. The [README](../README.md) describes the current system.

## Planned but not implemented

The following appear in the June 2026 documents but are not part of the code:

- User registration, login and student accounts. The application has no authentication; all attempts are stored in one shared history, and no student details are stored.
- The endpoints `/register`, `/login`, `/upload`, `/transcribe`, `/feedback` and `/report`.

| Document | Content that differs from the code |
|---|---|
| `Project Documentation/VBCUA_Report.pdf` | API table with the six endpoints above; database "stores student details"; testing section lists login |
| `Requirement Analysis/Solution Requirements_.pdf` | FR-1 User Authentication (registration form, email and password login) |
| `project planning/Project Planning Template.pdf` | Sprint stories USN-1 to USN-3 (registration and login) |
| `project Development phase/Functional & Performance Testing.pdf` | FT-01 User Login & Registration; response time of 2.7 s |
| `project Development phase/User Acceptance Testing.pdf` | Registration & Login test cases |
| `BrainsStroming and Ideantion/Brainstorming& idea presentation.pdf` | Login and speaking volume listed as ideas |

The implemented API has these routes:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/api/concepts` | Create a concept |
| GET | `/api/concepts` | List concepts |
| GET | `/api/concepts/{concept_id}` | Get one concept |
| POST | `/api/evaluate` | Upload audio and run the evaluation pipeline |
| GET | `/api/evaluations` | List past attempts |
| GET | `/api/evaluations/{eval_id}` | Get one attempt |
| GET | `/api/evaluations/{eval_id}/pdf` | Download the PDF report for an attempt |

## Test reports

The functional, performance and user acceptance test reports (12-06-2026) do not reflect the current code. The login and registration cases cover features that are planned but not implemented. The 2.7 s response time has not been re-measured with the current pipeline (local Whisper `small` on CPU). The current automated tests are in `tests/` and run with `python -m pytest`.

## Scoring and pipeline changes (October 2026)

Changes made on branch `research/week1-fixes`. The original formulas remain available as baseline `B0` (`method="b0"`) for comparison.

| Area | Change | Commit |
|---|---|---|
| Similarity | `(cos + 1) / 2` rescaling removed; similarity is `max(0, cos) × 100` | `e2a9dd5` |
| Transcription | Local Whisper `small` is the default; Gemini only as an opt-in fallback; ASR model and confidence stored per attempt | `95ce6c9` |
| LLM | `google-generativeai` and `gemini-1.5-flash` replaced by `google-genai` and `gemini-3.8-flash`, behind one client module | `92792e0` |
| Volume | RMS volume no longer affects fluency; used only for a "check your microphone" warning | `a5de403` |
| Fillers and pauses | Filler counts stored but marked experimental and not scored; pause ratio measured between first and last speech | `1ebb9cf` |
| Keywords | Whole-word matching (hyphen/space and simple plurals); no substring matches | `658f646`, `1e45b10` |
| Storage | Every sub-score, scoring method and feedback source stored per attempt | `6e7204d` |
| Frontend | Shows the stored backend scores and the evaluated concept's keywords | `9ad34fc` |
| Security | User-provided text escaped in the Streamlit HTML and the PDF report | `abf511f` |

Grade cut-offs (80 / 50) are unchanged and are not yet calibrated against human ratings.

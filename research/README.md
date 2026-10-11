# VBCUA research code

Code for the short research paper (RQ1: does rubric-based scoring agree with human ratings
better than the Sentence-BERT cosine score?). The app itself is in `backend/` and `web/`.

| Path | What it is |
|---|---|
| `evaluate.py` | Agreement metrics with 95% bootstrap CIs (1000 resamples, seed 42): QWK, Spearman, accuracy, confusion matrices; Pearson, Spearman, RMSE for numeric scores |
| `asag/` | Short-answer grading pilot: one loader per dataset, stratified split, cached Gemini calls, rubric generation, rubric grader, report |
| `asr/` | RQ2: synthetic speech (Piper TTS), cafeteria noise, Whisper transcription |
| `rubrics/` | Draft rubrics for six concepts (**DRAFT – NOT FACULTY REVIEWED**) and the review document generated from them |
| `results/` | Committed outputs: split IDs, generated rubrics, fitted cut-offs, metric tables. No dataset text or labels |
| `data/` | Datasets, per-answer predictions and the LLM cache. **Git-ignored, never committed** |

Install the extra packages with `pip install -r research/requirements.txt` (on top of
`backend/requirements.txt`). Run everything from the repository root.

## Datasets

Data files are never committed: neither dataset licence allows redistribution.

**Mohler (pilot dataset).** Mohler, Bunescu and Mihalcea (2011)[^mohler-hf]: 2,273 answers to
81 open-ended questions from an introductory data-structures course at the University of North
Texas, each graded 0–5 by two graders. The original release states no licence and asks users to
cite the paper below.

- Download `data/cleaned-oe-00001.parquet` (subset "cleaned", split "open_ended") into `research/data/mohler/`.
- `<STOP>` sentence markers are replaced by spaces. Closed-ended questions are excluded, as by the authors.
- **Primary metrics:** Pearson's r, Spearman's rho and RMSE against `score_avg` (0–5). Each method's score is mapped to 0–5 by a straight line fitted on dev only.
- **Secondary:** three labels: < 2.5 incorrect, 2.5 to < 4.5 partial, ≥ 4.5 correct (fixed before any results).
- **Human ceiling:** grader 1 vs grader 2 on the same test answers. The exams (E11, E12) were graded 0–10 per grader and are halved. For those exams the authors' `score_avg` is not exactly the mean of the halved grades (0.125–0.5 higher in 214 rows), so the ceiling compares the two graders with each other rather than with the average.

**EngSAF.** Aggarwal et al. (AIED 2025), available on request from the authors
(<https://github.com/dishankaggarwal/EngSAF>). It may only be used for the purpose stated in
the request, and redistribution is prohibited. Access has been requested. The loader expects the
official CSVs in `research/data/engsaf/`.

## Running the pilot

```
python -m research.asag.pilot split   --dataset mohler --data research/data/mohler   # fixed once, never re-drawn
python -m research.asag.pilot report  --dataset mohler --data research/data/mohler   # cosine methods (no API)
python -m research.asag.pilot plan    --dataset mohler --data research/data/mohler   # counts Gemini calls, makes none
python -m research.asag.pilot rubrics --dataset mohler --data research/data/mohler --max-calls N
python -m research.asag.pilot grade   --dataset mohler --data research/data/mohler --max-calls N
python -m research.asag.pilot report  --dataset mohler --data research/data/mohler   # adds the rubric grader
```

- **Split:** 300 answers stratified by label and then question (seed 42), then 100 dev / 200 test. Everything that is fitted uses dev only, and every metric is reported on test only.
- **Calls are capped:** `--max-calls` limits the new Gemini calls per run, and the default 0 makes none. Every answer is cached in `research/data/llm_cache/`, keyed by (model, prompt hash), at temperature 0.
- **Free-tier keys:** run `rubrics` / `grade` once a day with `--max-calls` at or below the daily quota. Each run resumes where the last one stopped. A run also stops cleanly when the daily quota is used up.

## Citations

Mohler, M., Bunescu, R., and Mihalcea, R. (2011). Learning to Grade Short Answer Questions using
Semantic Similarity Measures and Dependency Graph Alignments. In *Proceedings of the 49th Annual
Meeting of the Association for Computational Linguistics: Human Language Technologies*,
pp. 752–762, Portland, Oregon. <https://aclanthology.org/P11-1076/>

Aggarwal, D., Sil, P., Raman, B., and Bhattacharyya, P. (2025). "I understand why I got this
grade": Automatic Short Answer Grading (ASAG) with Feedback. In *Artificial Intelligence in
Education (AIED 2025)*. (Cite only if EngSAF is used.)

### RQ2 synthetic speech (tools and data)

Audio is generated and kept in the git-ignored `research/data/tts/` and `research/data/noise/`;
none of it is redistributed. Used for non-commercial academic research only.

- **Piper** text-to-speech, `piper-tts` 1.8.0 (Open Home Foundation, <https://github.com/OHF-Voice/piper1-gpl>),
  GPL-3.0-or-later. Voice models from <https://huggingface.co/rhasspy/piper-voices>:
  - `en_US-l2arctic-medium`: trained on L2-ARCTIC, licence CC BY-NC 4.0. One of its Hindi-L1
    speakers (ASI, RRBI, SVBI or TNI) is the Indian-English voice.
  - `en_GB-alba-medium`: trained on the Alba (Scottish English) corpus,
    <https://datashare.ed.ac.uk/handle/10283/3270>, licence CC BY 4.0.
  - `en_US-ryan-medium`: trained on RyanSpeech, <https://www.kaggle.com/datasets/roholazandie/ryanspeech>,
    licence CC BY-NC-SA 4.0.
- **L2-ARCTIC**: Zhao, G., Sonsaat, S., Silpachai, A., Lucic, I., Chukharev-Hudilainen, E., Levis, J.,
  and Gutierrez-Osuna, R. (2018). L2-ARCTIC: A Non-native English Speech Corpus. In *Proc. Interspeech
  2018*, pp. 2783–2787. <https://psi.engr.tamu.edu/l2-arctic-corpus/>, CC BY-NC 4.0.
- **RyanSpeech**: Zandie, R., Mahoor, M. H., Madsen, J., and Emamian, E. S. (2021). RyanSpeech: A Corpus
  for Conversational Text-to-Speech Synthesis. In *Proc. Interspeech 2021*.
- **Alba corpus**: Centre for Speech Technology Research, University of Edinburgh, *Alba speech corpus*
  (Edinburgh DataShare, <https://datashare.ed.ac.uk/handle/10283/3270>), CC BY 4.0.
- **DEMAND** noise (cafeteria recording `PCAFETER`, 16 kHz, channel 1): Thiemann, J., Ito, N., and
  Vincent, E. (2013). The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND): A database
  of multichannel environmental noise recordings. *Proc. Meetings on Acoustics* 19, 035081.
  <https://zenodo.org/records/1227121>, CC BY 4.0.
- **Whisper** (local `openai-whisper`, models tiny / base / small), MIT licence: Radford, A., Kim, J. W.,
  Xu, T., Brockman, G., McLeavey, C., and Sutskever, I. (2023). Robust Speech Recognition via Large-Scale
  Weak Supervision. In *Proc. ICML 2023*.

Check the exact author lists and page numbers against the publishers' pages before submission.

[^mohler-hf]: Obtained from the Hugging Face conversion by N. Kazi,
<https://huggingface.co/datasets/nkazi/MohlerASAG> (subset "cleaned", split "open_ended"),
which keeps the authors' identifiers, questions, answers and grades.

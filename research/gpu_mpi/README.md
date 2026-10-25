# GPU/MPI: with documentation vs closed-book (Oct 25)

GPU/MPI questions (d1) are the weakest set. On the part about a cluster whose documentation is held out, the search
returns unrelated Slurm pages and the model follows them, while closed-book answers often do better.

| Script | What it does |
|---|---|
| `gate27.sh` | Round 27 answers every test question closed-book (1,536 tokens); the three judges score it against the same reference |
| `gate_analysis.py` | Per question: score with excerpts vs closed-book, the ceiling (better of the two), and word-coverage switching rules |
| `self_select.py`, `self_select.sbatch` | The model itself picks between its two answers (question and answers only, both orders) |
| `sel_analysis.py` | Scores those picks with the judged results |

Results (round 27, 3-judge majority): d1 0.700 with excerpts, 0.663 closed-book, **0.804 for the better of the two**
(v3 0.885 → 0.939). A word-coverage rule (< 40% of the question's words in the excerpts → closed-book) gives d1
+0.019; the model choosing for itself gives +0.007 (it mostly keeps its first answer). `research/round50/` trains the
choice into the model instead.

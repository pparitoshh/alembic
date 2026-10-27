# Mentor use cases

Real Slurm tasks from users of LUMI, Leonardo and other EuroHPC clusters. Our mentor supplies them.

**Do not use these files for training, gold examples, preference data or retrieval.** They are held-out test
data, the same as `data/eval/`. Copy a case into a training set only when the mentor and the team agree.

## Folders

| Folder | Content |
|---|---|
| `lumi/` | Use cases from LUMI users |
| `leonardo/` | Use cases from Leonardo users |
| `other/` | Use cases from other clusters (write the cluster name in the `cluster` field) |

## How to add a use case

1. Open the folder for the cluster.
2. Add one line to `usecases.jsonl` in that folder. Use the format below.
3. If the case has a full batch script or a log, put it in a file next to `usecases.jsonl`.
4. Write the file name in the `files` field.
5. Remove user names, project IDs, paths with personal names and all secrets before you commit.

A free-text Markdown file is also acceptable. We will convert it to the format below.

## Format (one JSON object on each line)

```json
{
  "id": "lumi-001",
  "cluster": "lumi",
  "user_request": "My PyTorch job on 4 GPUs stays PENDING for 2 days. Why?",
  "context": "Partition small-g, 4 GCDs, --time=48:00:00",
  "expected_actions": ["job_status", "partition_info"],
  "expected_answer": "MaxTime on small-g is 3 days, but the reason is QOSMaxGRESPerUser ...",
  "must_not": ["cancel the job without asking"],
  "files": ["lumi-001.sbatch"],
  "source": "user ticket, anonymised",
  "difficulty": "medium"
}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | `<cluster>-<number>`, unique |
| `cluster` | yes | `lumi`, `leonardo` or another cluster name |
| `user_request` | yes | What the user asked, in the user's words |
| `context` | no | Facts the assistant can know: partition, script parts, error text |
| `expected_actions` | no | Tools or commands a good assistant uses (`job_status`, `sacct`, ...) |
| `expected_answer` | no | The correct answer or the important points of it |
| `must_not` | no | Wrong or dangerous actions |
| `files` | no | Scripts or logs in this folder |
| `source` | no | Where the case comes from |
| `difficulty` | no | `easy`, `medium` or `hard` |

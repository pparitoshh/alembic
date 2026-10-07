"""Conservative target check for job_status; no model guesses or hidden gold values."""
import re
from .tools import ToolError, parse_arguments

# An unrelated resource count is not a job identifier. This bounded grammar may reject
# unusual phrasing; it never treats source text/gold as user authorization.
_EXPLICIT = re.compile(r"\bjob(?:\s*(?:id|number))?\s*[:#=]?\s*[`\"']?([0-9]+(?:_[0-9]+)?)(?![0-9_])", re.I)
_STATUS = re.compile(r"\b(?:state|status)\s+of\s*[`\"']?([0-9]+(?:_[0-9]+)?)(?![0-9_])", re.I)
_IS = re.compile(r"\bIs\s+([0-9]+(?:_[0-9]+)?)\s+(?:still\s+)?(?:running|pending|finished|completed|failed)\b", re.I)
_ACTION = re.compile(r"\b(?:cancel|scancel)\s+([0-9]+(?:_[0-9]+)?)(?![0-9_])", re.I)
_USAGE = re.compile(r"\b(?:resource\s+usage|logs?)\s+for\s+([0-9]+(?:_[0-9]+)?)(?![0-9_])", re.I)
_RESULT = re.compile(r'"job_id"\s*:\s*"([0-9]+(?:_[0-9]+)?)"')


def target_error(call, question, results=()):
    if call.get('name') not in {'job_status', 'job_accounting', 'read_job_log', 'cancel_job'}:
        return None
    try:
        args = parse_arguments(call.get('arguments', {}))
    except ToolError:
        return None  # normal schema validation reports malformed arguments
    job_id = args.get('job_id')
    if not isinstance(job_id, str):
        return None
    known = set(_EXPLICIT.findall(question)) | set(_STATUS.findall(question)) | set(_IS.findall(question)) | set(_ACTION.findall(question)) | set(_USAGE.findall(question)) | set(_RESULT.findall(question))
    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get('job_id'), str) and 'error' not in value:
                known.add(value['job_id'])
            for child in value.values():collect(child)
        elif isinstance(value, list):
            for child in value:collect(child)
    for result in results:collect(result)
    if job_id not in known:
        return f"{call['name']} target {job_id!r} is not an explicitly identified job in the user request or preceding tool results; ask for the job ID"
    return None


WORKFLOW_VERSION = "job-status-evidence-v1"
WORKFLOW_RULES = """
Bounded job-status workflow:
- This workflow answers one identified job-status request or a general question about status fields.
- If the target job ID is missing or ambiguous, ask for the missing identifier before any lookup.
  Do not list the queue merely to work around a missing ID; queue discovery needs an explicit user request.
- After a status result, summarize only relevant returned fields and limitations explicitly established by
  the supplied source. Do not enumerate possible application, launcher, runtime or resource failure causes.
  A state and exit code do not establish those causes. It is enough to state what cannot be determined.
- This workflow's source supplies tool contracts, not shell-command syntax. Do not offer a shell command,
  command example, or script. If an appropriate next diagnostic tool is mentioned, use only its documented
  name/capability and do not claim to have run it. Accounting fields are not stderr log lines.
- A not-found result means only absence from this simulated catalog. Do not infer prior existence,
  removal, completion or cancellation from that result.
"""
# Hard check for the observed unsupported-command defect. This does not prove semantic grounding.
_SHELL_BLOCK = re.compile(r"```(?:bash|sh|shell|console)\b", re.I)
_CLUSTER_COMMAND = re.compile(r"(?:^|[\n`])\s*(?:sacct|squeue|scontrol|scancel|sbatch|srun)\s+[-\w]", re.I)


def response_issues(row, answer):
    workflow = row.get('workflow_version')
    if workflow is None:
        return []
    if workflow != WORKFLOW_VERSION:
        return [f'unsupported workflow version {workflow!r}']
    if _SHELL_BLOCK.search(answer) or _CLUSTER_COMMAND.search(answer):
        return ['job-status workflow supplied an unsupported shell command; its source provides tool contracts only']
    return []

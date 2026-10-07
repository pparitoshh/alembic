"""Conservative target check for job_status; no model guesses or hidden gold values."""
import re
from dataclasses import asdict, dataclass
from .tools import ToolError, parse_arguments

# An unrelated resource count is not a job identifier. This bounded grammar may reject
# unusual phrasing; it never treats source text/gold as user authorization.
_EXPLICIT = re.compile(r"\bjob(?:\s*(?:id|number))?\s*[:#=]?\s*[`\"']?([0-9]+(?:_[0-9]+)?)(?![0-9_])", re.I)
_STATUS = re.compile(r"\b(?:state|status)\s+of\s*[`\"']?([0-9]+(?:_[0-9]+)?)(?![0-9_])", re.I)
_IS = re.compile(r"\bIs\s+([0-9]+(?:_[0-9]+)?)\s+(?:still\s+)?(?:running|pending|finished|completed|failed)\b", re.I)
_ACTION = re.compile(r"\b(?:cancel|scancel)\s+([0-9]+(?:_[0-9]+)?)(?![0-9_])", re.I)
_USAGE = re.compile(r"\b(?:resource\s+usage|logs?)\s+for\s+([0-9]+(?:_[0-9]+)?)(?![0-9_])", re.I)
_RESULT = re.compile(r'"job_id"\s*:\s*"([0-9]+(?:_[0-9]+)?)"')


def identified_jobs(question, results=()):
    """Conservative visible identifiers, never reference answers or expected modes."""
    known = set().union(*(set(p.findall(question)) for p in (_EXPLICIT, _STATUS, _IS, _ACTION, _USAGE, _RESULT)))
    def collect(value):
        if isinstance(value, dict):
            if 'error' in value or value.get('found') is False:
                return
            if isinstance(value.get('job_id'), str):
                known.add(value['job_id'])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    for result in results:
        collect(result)
    return known


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
    known = identified_jobs(question, results)
    if job_id not in known:
        return f"{call['name']} target {job_id!r} is not an explicitly identified job in the user request or preceding tool results; ask for the job ID"
    return None


LEGACY_WORKFLOW_VERSION = "job-status-evidence-v1"
WORKFLOW_VERSION = "job-status-evidence-v2"
POLICY_VERSION = "job-status-policy-v2"
_READ_ONLY = {'job_status', 'list_queue', 'read_job_log', 'job_accounting', 'gpu_availability', 'partition_info'}
_DISCOVERY = re.compile(r'\b(?:list|show|find|search)\b[^.!?]*\b(?:jobs|queue)\b', re.I)
_RESOURCES = re.compile(r'\b(?:show|list|check|look up)\s+(?:the\s+)?(?:current\s+)?(?:free|available|partition)\b', re.I)
_NO_LOOKUP = re.compile(r"\b(?:do not|don't|never)\s+(?:use\s+(?:any\s+)?tools|call|query|look\s*up|inspect|list|show|check|find|search)|\bask\b[^.!?]*\bbefore\b[^.!?]*(?:inspect|lookup|look\s*up|tool|query|list|show|check|find|search)", re.I)
_SELECT_SINGLE_RESULT = re.compile(r'\b(?:check|inspect)\b[^.!?]*\b(?:only|single)\b[^.!?]*\b(?:result|job)\b', re.I)


@dataclass(frozen=True)
class JobStatusPolicy:
    """Deployment-visible permission policy; no test labels or desired answers.

    This bounded workflow handles one selected job. Default generation without this
    policy retains its existing broader tools. Explicit queue requests remain valid.
    """
    discovery: str = 'clarify_first'
    version: str = POLICY_VERSION

    def __post_init__(self):
        if self.version != POLICY_VERSION or self.discovery not in {'clarify_first', 'allow_readonly'}:
            raise ValueError('unknown job-status application policy')

    def as_dict(self):
        return asdict(self)

    def instructions(self):
        discovery = ('Ask for the target before lookup; queue discovery requires an explicit user request.'
                     if self.discovery == 'clarify_first' else
                     'Read-only queue discovery may precede clarification. A discovered candidate is not a selected target.')
        return f'\nApplication policy {self.version}: {discovery} Only read-only diagnostics are permitted. Explicit queue/resource-list requests need no job ID. Ask which job when selection is ambiguous. A sole queue candidate is not automatically the requested job; inspect it only if the user explicitly requested inspection of the only result. Explicit user instructions to ask before tools take precedence.\n'


def policy_error(call, question, results, policy):
    """Check a call against visible user intent and already executed earlier turns."""
    if not isinstance(call, dict) or not isinstance(call.get('name'), str):
        return 'Malformed tool call; no lookup or action was executed.'
    if _NO_LOOKUP.search(question):
        return 'The user requested clarification before lookup or prohibited tool use.'
    name = call.get('name')
    if name not in _READ_ONLY:
        return 'This application policy permits only read-only job diagnostics.'
    if name in {'gpu_availability', 'partition_info'}:
        return None if _RESOURCES.search(question) else 'A live resource lookup requires an explicit resource request.'
    targets = identified_jobs(question)
    if name == 'list_queue':
        if _DISCOVERY.search(question) or policy.discovery == 'allow_readonly':
            return None
        return 'Queue discovery was not requested; ask for the missing job identification before lookup.'
    if len(targets) > 1:
        return 'More than one job is identified; ask which target to inspect.'
    target_issue = target_error(call, question, results)
    if target_issue:
        return target_issue
    try:
        job_id = parse_arguments(call.get('arguments', {})).get('job_id')
    except ToolError:
        return None  # schema validation reports malformed arguments
    if targets:
        return None if job_id in targets else 'The call does not target the job selected by the user.'
    discovered = identified_jobs('', results)
    if len(discovered) == 1 and _SELECT_SINGLE_RESULT.search(question) and job_id in discovered:
        return None
    return 'A discovered job is not a selected target; ask the user to identify the intended job.'


def blocked_batch(calls, question, results, policy):
    """Validate the entire authored batch before executing any member."""
    errors = [policy_error(call, question, results, policy) for call in calls]
    if not any(errors):
        return []
    return [error or 'Batch not executed because another simultaneous call violated policy.' for error in errors]


def guard_result(reason, policy):
    return {'error': reason, 'guard_blocked': True, 'policy_version': policy.version}


def clarification_response(question):
    if len(identified_jobs(question)) > 1:
        return 'Which job ID should I inspect?'
    return 'I have not performed that lookup or action. Please identify the intended job and the permitted read-only check.'
WORKFLOW_RULES = """
Bounded job-status workflow:
- This workflow answers one identified job-status request or a general question about status fields.
- Follow the explicit application policy for discovery and clarification. Never guess a target.
- After a status result, summarize only relevant returned fields and limitations explicitly established by
  the supplied source. A state and exit code do not establish a failure cause. A hypothetical example
  still makes technical claims: include it only if the source supports those claims, and label it as hypothetical.
  Do not invent possible causes merely to make an explanation more detailed.
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
    if workflow not in {LEGACY_WORKFLOW_VERSION, WORKFLOW_VERSION}:
        return [f'unsupported workflow version {workflow!r}']
    if _SHELL_BLOCK.search(answer) or _CLUSTER_COMMAND.search(answer):
        return ['job-status workflow supplied an unsupported shell command; its source provides tool contracts only']
    return []

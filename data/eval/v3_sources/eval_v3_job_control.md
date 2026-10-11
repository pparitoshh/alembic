# Evaluation source bundle: job control

## Source excerpt: slurm_man_scontrol#9

**reconfigure**
Instruct all slurmctld and slurmd daemons to re-read the configuration file.
This mechanism can be used to modify configuration parameters set in
slurm.conf(5) without interrupting running jobs.
Starting in 23.11, this command operates by creating new processes for the
daemons, then passing control to the new processes when or if they
start up successfully. This allows it to gracefully catch configuration
problems and keep running with the previous configuration if there is a problem.
This will not be able to change the daemons' listening TCP port settings
or authentication mechanism.

**release <job_list>**
Release a previously held job to begin execution.
The job_list argument is a space separated list of job IDs (numeric or SLUID)
OR "jobname=" with the job's name, which will attempt to release all jobs
having that name.
Note that a privileged user may release an admin-hold or a user-hold.
An unprivileged job owner or coordinator may only release a user-hold.
Also see hold.

**State=SpecialExit**
The "SpecialExit" keyword specifies that the job
has to be put in a special state JOB_SPECIAL_EXIT.
The "scontrol show job" command will display the JobState as
SPECIAL_EXIT, while the "squeue" command as SE.

## Source excerpt: slurm_man_scontrol#18

**update <SPECIFICATION>**
Update job, step, node, partition, or reservation configuration per
the supplied specification. SPECIFICATION is in the same format as the
Slurm configuration file and the output of the show command described
above. It may be desirable to execute the show command (described above)
on the specific entity you want to update, then use cut-and-paste tools to
enter updated configuration values to the update. Note that while most
configuration values can be changed using this command, not all can be changed
using this mechanism. In particular, the hardware configuration of a node or
the physical addition or removal of nodes from the cluster may only be
accomplished through editing the Slurm configuration file and executing
the reconfigure command (described above).

**update <SuspendExc*>[=|+=|-=]<LIST>**
Update SuspendExcNodes, SuspendExcParts, or SuspendExcStates.
<LIST> is either a NodeList, list of partitions, or list of node states
respectively. Use +=/-= to add/remove nodes, partitions, or states to/from the
currently configured list. Use = to replace the current list.
SuspendExcNodes does not support "+="/"-=" when the ":" option is used,
however, direct assignment "=" is always supported.
Consider using "scontrol show config | grep SuspendExc" to see current state of
these settings.

**version**
Display the version number of scontrol being executed.

## Source excerpt: slurm_man_scontrol#21

An unprivileged user may modify their own jobs subject to certain restrictions,
as well as jobs and steps in any accounts where they have Coordinator
permissions. Most of these fields can only be updated on pending jobs.
Some may be updated on running or suspended jobs.
A privileged user can modify any jobs without policy restrictions. For instance,
if an administrator changes the QOS on a pending job, certain limits such as the
TimeLimit will not be changed automatically as changes made by privileged users
are allowed to violate these restrictions.

**JobName=<name>**
Identify the name of jobs to be modified or set the job's name to the
specified value.
When used to identify jobs to be modified, all jobs belonging to all users
are modified unless the UserID option is used to identify a specific user.
Either JobId or JobName is required.

**Account=<account>**
Account name to be changed for this job's resource use.
Value may be cleared with blank data value, "Account=".

**AdminComment=<spec>**
Arbitrary descriptive string. Can only be set by a Slurm administrator.

**BurstBuffer=<spec>**
Burst buffer specification to be changed for this job's resource use.
Value may be cleared with blank data value, "BurstBuffer=".
Format is burst buffer plugin specific.
Only a privileged user can change this parameter.

**Clusters=<spec>**
Specifies the clusters that the federated job can run on.

## Source excerpt: slurm_man_scontrol#32

**StdErr=<filepath>**
Set the batch job's stderr file path.
Value may be reset to job default with blank data value, "StdErr=".
NOTE: By default, StdErr will be merged into StdOut.

**StdIn=<filepath>**
Set the batch job's stdin file path.
Value may be reset to job default with blank data value, "StdIn=".
NOTE: By default, StdIn will be '/dev/null'.

**StdOut=<filepath>**
Set the batch job's stdout file path.
Value may be reset to job default with blank data value, "StdOut=".
NOTE: By default, StdOut will be based on the JobId.

**StartTime=<time_spec>**
Set the job's earliest initiation time.
It accepts times of the form HH:MM:SS to run a job at
a specific time of day (seconds are optional).
(If that time is already past, the next day is assumed.)
You may also specify midnight, noon, elevenses (11 AM),
fika (3 PM) or teatime (4 PM) and you can have a time-of-day
suffixed with AM or PM for running in the morning or the evening.
You can also say what day the job will be run, by specifying
a date of the form MMDDYY or MM/DD/YY or MM.DD.YY,
or a date and time as YYYY-MM-DD[THH:MM[:SS]]. You can also
give times like now + count time-units, where the time-units
can be seconds (default), minutes, hours, days,
or weeks. The keywords today and tomorrow may also be used.

## Source excerpt: slurm_man_scontrol#37

**TimeLimit=<time>**
The job's time limit.
Output format is [days-]hours:minutes:seconds or "UNLIMITED".
Input format (for update command) set is minutes, minutes:seconds,
hours:minutes:seconds, days-hours, days-hours:minutes or
days-hours:minutes:seconds.
Time resolution is one minute and second values are rounded up to
the next minute.
If changing the time limit of a step, either specify a new time limit value or
precede the time with a "+" or "-" to increment or decrement the current
time limit (e.g. "TimeLimit=+30"). In order to increment or decrement the
current time limit, the StepId specification must precede the
TimeLimit specification.

## Nodes - Specifications For Create Command

Provide the same NodeName configuration as found in the slurm.conf. See
slurm.conf man page for details. Only State=CLOUD and State=FUTURE nodes are
allowed.

## Nodes - Specifications For Update Command

**NodeName=<name>**
Identify the node(s) to be updated. Multiple node names may be specified using
simple node range expressions (e.g. "lx[10-20]"). Nodesets can also be
specified by themselves or mixed with node range expressions, using a comma as
a list separator. If the keyword "ALL" is specified alone, then the update
will be attempted against all the nodes in the local cluster. This specification
is required.

## Source excerpt: slurm_man_sbatch#65

**--time-min=<time>**
Set a minimum time limit on the job allocation.
If specified, the job may have its --time limit lowered to a value
no lower than --time-min if doing so permits the job to begin
execution earlier than otherwise possible.
The job's time limit will not be changed after the job is allocated resources.
This is performed by a backfill scheduling algorithm to allocate resources
otherwise reserved for higher priority jobs.
Acceptable time formats include "minutes", "minutes:seconds",
"hours:minutes:seconds", "days-hours", "days-hours:minutes" and
"days-hours:minutes:seconds".

**--tmp=<size>[units]**
Specify a minimum amount of temporary disk space per node.
Default units are mebibytes.
Different units can be specified using the suffix [K|M|G|T].

**--tres-bind=<tres>:[verbose,]<type>[+<tres>:**
[verbose,]<type>...]
Specify a list of tres with their task binding options. Currently gres are the
only supported tres for this options. Specify gres as "gres/<gres_name>"
(e.g. gres/gpu)
Example: --tres-bind=gres/gpu:verbose,map:0,1,2,3+gres/nic:closest
By default, most tres are not bound to individual tasks
Supported binding type options for gres:

**closest**
Bind each task to the gres(s) which are closest.
In a NUMA environment, each task may be bound to more than one gres (i.e.
all gres in that NUMA environment).

## Source excerpt: upstream_scontrol_2f1db40fc86d

**hold <job_list>**
Prevent a pending job from being started (sets its priority to 0).
The job_list argument is a space separated list of job IDs (numeric or SLUID)
OR "jobname=" with the job's name, which will attempt to hold all jobs having
that name.
Only a privileged user, account coordinator, or job owner may hold jobs. If
issued by a privileged user an admin-hold will be placed on the job, otherwise
a user-hold will be placed on the job. The hold type determines which users may
remove the hold with the release command (also see uhold).
Additionally, attempting to hold a running job will not suspend or cancel
it. But, it will set the job priority to 0 and update the job reason field,
which would hold the job if it was requeued at a later time.

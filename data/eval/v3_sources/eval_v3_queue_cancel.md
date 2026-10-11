# Evaluation source bundle: queue cancel

## Source excerpt: slurm_man_squeue#0

## Name

squeue - view information about jobs located in the Slurm scheduling queue.

## Synopsis

squeue [OPTIONS...]

## Description

squeue is used to view job and job step information for jobs managed by
Slurm.

## Options

**-A, --account=<account_list>**
Specify the accounts of the jobs to view. Accepts a comma separated
list of account names. This has no effect when listing job steps.

**-a, --all**
Display information about jobs and job steps in all partitions.
This causes information to be displayed about partitions that are configured as
hidden, partitions that are unavailable to a user's group, and federated jobs
that are in a "revoked" state.

**-M, --clusters=<cluster_name>**
Clusters to issue commands to. Multiple cluster names may be comma separated.
A value of 'all' will query to run on all clusters.
Note that the slurmdbd must be up for this option to work properly, unless
running in a federation with either FederationParameters=fed_display
configured or the --federation option set.
This option implicitly sets the --local option.

**--expand-patterns**
Expand any filename patterns from in \f3StdOut, \f3StdErr and \f3StdIn.
Fields that map to a range of values will use the first value of the range. For
example "%t" for task id will be replaced by "0".

**--federation**
Show jobs from the federation if a member of one.

## Source excerpt: slurm_man_squeue#1

**-o, --format=<output_format>**
Specify the information to be displayed, its size and position
(right or left justified).
Also see the -O, --Format=<output_format>
option described below (which supports less flexibility in formatting, but
supports access to all fields).
If the command is executed in a federated cluster environment and information
about more than one cluster is to be displayed and the -h, --noheader
option is used, then the cluster name will be displayed before the default
output formats shown below.
The default formats with various options are:

**default**
"%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

**-l, --long**
"%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"

**-s, --steps**
"%.15i %.8j %.9P %.8u %.9M %N"
The format of each field is "%[[.]size]type[suffix]"

**size**
Minimum field size. If no size is specified, whatever is needed to print the
information will be used.

**.**
Indicates the output should be right justified and size must be specified.
By default output is left justified.

**suffix**
Arbitrary string to append to the end of the field.
Note that many of these type specifications are valid
only for jobs while others are valid only for job steps.
Valid type specifications include:

**%all**
Print all fields available for this data type with a vertical bar separating
each field.

**%a**
Account associated with the job.
(Valid for jobs only)

## Source excerpt: slurm_man_squeue#17

**-i, --iterate=<seconds>**
Repeatedly gather and report the requested information at the interval
specified (in seconds).
By default, prints a time stamp with the header.

**-j, --jobs[=<job_id_list>]**
Specify a comma separated list of job IDs to display. Defaults to all jobs.
The --jobs=<job_id_list> option may be used in conjunction with
the --steps option to print step information about specific jobs.
Note: If a list of job IDs is provided, the jobs are displayed even if
they are on hidden partitions. Since this option's argument is optional,
for proper parsing the single letter option must be followed immediately
with the value and not include a space between them. For example "-j1008"
and not "-j 1008".
The job ID format is "job_id[_array_id]". A SLUID may also be used in place
of a numeric job ID.
Performance of the command can be measurably improved for systems with large
numbers of jobs when a single job ID is specified.
By default, this field size will be limited to 64 bytes.
Use the environment variable SLURM_BITSTR_LEN to specify larger field sizes.

**\f3--json, \f3--json=list, \f3--json=<data_parser>**
Dump information as JSON using the default data_parser plugin or explicit
data_parser with parameters. All information is dumped, even if it would
normally not be. Sorting and formatting arguments passed to other options are
ignored; however, most filtering arguments are still used.

## Source excerpt: slurm_man_squeue#21

**-t, --states=<state_list>**
Specify the states of jobs to view. Accepts a comma separated list of
state names or "all". If "all" is specified then jobs of all states will be
reported. If no state is specified then pending, running, and completing
jobs are reported. See the JOB STATE CODES section below for a list of
valid states. Both extended and compact forms are valid.
Note the <state_list> supplied is case insensitive ("pd" and "PD" are
equivalent).

**-s, --steps[=<step_list>]**
Specify the job steps to view. This flag indicates that a comma separated list
of job steps to view follows without an equal sign (see examples).
The job step format is "job_id[_array_id].step_id". Defaults to all job
steps. Since this option's argument is optional, for proper parsing
the single letter option must be followed immediately with the value
and not include a space between them. For example "-s1008.0" and not
"-s 1008.0".

**--usage**
Print a brief help message listing the squeue options.

**-u, --user=<user_list>**
Request jobs or job steps from a comma separated list of users.
The list can consist of user names or user id numbers.
Performance of the command can be measurably improved for systems with large
numbers of jobs when a single user is specified.

**-v, --verbose**
Report details of squeues actions.

**-V , --version**
Print version information and exit.

## Source excerpt: slurm_man_squeue#23

**DependencyNeverSatisfied**
This job has a dependency on another job that will never be satisfied.

**InactiveLimit**
The job reached the system InactiveLimit.

**InvalidAccount**
The job's account is invalid.

**InvalidQOS**
The job's QOS is invalid.

**JobHeldAdmin**
The job is held by a system administrator.

**JobHeldUser**
The job is held by the user.

**JobLaunchFailure**
The job could not be launched.
This may be due to a file system problem, invalid program name, etc.

**Licenses**
The job is waiting for a license.

**NodeDown**
A node required by the job is down.

**NonZeroExitCode**
The job terminated with a non-zero exit code.

**PartitionDown**
The partition required by this job is in a DOWN state.

**PartitionInactive**
The partition required by this job is in an Inactive state and not able to
start jobs.

**PartitionNodeLimit**
The number of nodes required by this job is outside of its partition's current
limits.
Can also indicate that required nodes are DOWN or DRAINED.

**PartitionTimeLimit**
The job's time limit exceeds its partition's current time limit.

**Priority**
One or more higher priority jobs exist for this partition or advanced reservation.

**Prolog**
Its Prolog program is still running.

**QOSGrp*Limit**
The job's QOS has reached an aggregate limit on some resource.

**QOSJobLimit**
The job's QOS has reached its maximum job count.

**QOSMax*Limit**
The job requests a resource that violates a per-job limit on the requested
QOS.

## Source excerpt: slurm_man_squeue#25

**CD  COMPLETED**
Job has terminated all processes on all nodes with an exit code of zero.

**CF  CONFIGURING**
Job has been allocated resources, but are waiting for them to become ready for use
(e.g. booting).

**CG  COMPLETING**
Job is in the process of completing. Some processes on some nodes may still be active.

**DL  DEADLINE**
Job terminated on deadline.

**F   FAILED**
Job terminated with non-zero exit code or other failure condition.

**NF  NODE_FAIL**
Job terminated due to failure of one or more allocated nodes.

**OOM OUT_OF_MEMORY**
Job experienced out of memory error.

**PD  PENDING**
Job is awaiting resource allocation.

**R   RUNNING**
Job currently has an allocation.

**RD  RESV_DEL_HOLD**
Job is being held after requested reservation was deleted.

**RS  RESIZING**
Job is about to change size.

**RV  REVOKED**
Sibling was removed from cluster due to other cluster starting the job.

**SI  SIGNALING**
Job is being signaled.

**SO  STAGE_OUT**
Job is staging out files.

**S   SUSPENDED**
Job has an allocation, but execution has been suspended and CPUs have been
released for other jobs.

**TO  TIMEOUT**
Job terminated upon reaching its time limit.

## Performance

Executing squeue sends a remote procedure call to slurmctld. If
enough calls from squeue or other Slurm client commands that send remote
procedure calls to the slurmctld daemon come in at once, it can result in
a degradation of performance of the slurmctld daemon, possibly resulting
in a denial of service.

## Source excerpt: slurm_man_scancel#0

## Name

## Synopsis

scancel [OPTIONS...] [job_id[_array_id][.step_id]] [job_id[_array_id][.step_id]...]

## Description

## Options

**-A, --account=account**
Restrict the scancel operation to jobs under this charge account.

**--admin-comment=comment**
Set the AdminComment on the job while canceling jobs. User must have
Administrator privileges on the system.

**-M, --clusters=<string>**
Cluster to issue commands to. Implies --ctld.
Note that the slurmdbd must be up for this option to work properly, unless
running in a federation with FederationParameters=fed_display configured.

**-c, --cron**
Confirm request to cancel a job submitted by scrontab. This option only has
effect with the "explicit_scancel" option is set in ScronParameters.

**--help**
Print a help message describing all scancel options.

**-H, --hurry**
Do not stage out any burst buffer data.

**-i, --interactive**
Interactive mode. Confirm each job_id.step_id before performing the cancel operation.

**-n, --jobname=job_name, --name=job_name**
Restrict the scancel operation to jobs with this job name.

**--me**
Restrict the scancel operation to jobs owned by the current user.

**-w, --nodelist=host1,host2,...**
Cancel any jobs using any of the given hosts. The list may be specified as
a comma-separated list of hosts, a range of hosts (host[1-5,7,...] for
example), or a filename. The host list will be assumed to be a filename only
if it contains a "/" character.

## Source excerpt: slurm_man_scancel#1

**-p, --partition=partition_name**
Restrict the scancel operation to jobs in this partition.

**-q, --qos=qos**
Restrict the scancel operation to jobs with this quality of service.

**-Q, --quiet**
Do not report an error if the specified job is already completed.
This option is incompatible with the --verbose option.

**-R, --reservation=reservation_name**
Restrict the scancel operation to jobs with this reservation name.

**--sibling=cluster_name**
Remove an active sibling job from a federated job.

**-t, --state=job_state_name**
Restrict the scancel operation to jobs in this
state. job_state_name may have a value of either "PENDING",
"RUNNING" or "SUSPENDED".

**--usage**
Print a brief help message listing the scancel options.

**-u, --user=user_name**
Restrict the scancel operation to jobs owned by the given user.

**-v, --verbose**
Print additional logging. Multiple v's increase logging detail.
This option is incompatible with the --quiet option.

**-V, --version**
Print the version number of the scancel command.

**--wckey=wckey**
Restrict the scancel operation to jobs using this workload
characterization key.

##

ARGUMENTS

**job_id**
The Slurm job ID to be signaled. A SLUID may also be used in place of a
numeric job ID.

## Performance

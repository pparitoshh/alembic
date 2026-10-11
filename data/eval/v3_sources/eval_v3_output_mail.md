# Evaluation source bundle: output mail

## Source excerpt: slurm_man_sbatch#1

By default both standard output and standard error are directed to a file of
the name "slurm-%j.out", where the "%j" is replaced with the job allocation
number. The file will be generated on the first node of the job allocation.
Other than the batch script itself, Slurm does no movement of user files.

When the job allocation is granted for the batch job, Slurm runs a single copy
of the batch script on one of the allocated nodes. The batch host is typically
the first node in the allocation, but may be placed elsewhere due to the
--batch option.

The following document describes the influence of various options on the
allocation of cpus to jobs and tasks.

https://slurm.schedmd.com/cpu_management.html

## Return Value

sbatch will return 0 on success or error code on failure.

## Script Path Resolution

The batch script is resolved in the following order:

1. If script starts with ".", then path is constructed as:
current working directory / script

2. If script starts with a "/", then path is considered absolute.

3. If script is in current working directory.

4. If script can be resolved through PATH. See path_resolution(7).

Current working directory is the calling process working directory unless the
--chdir argument is passed, which will override the current working
directory.

## Options

## Source excerpt: slurm_man_sbatch#37

**-i, --input=<filename_pattern>**
Instruct Slurm to connect the batch script's standard input
directly to the file name specified in the "filename pattern".
By default, "/dev/null" is open on the batch script's standard input and both
standard output and standard error are directed to a file of the name
"slurm-%j.out", where the "%j" is replaced with the job allocation number, as
described below in the filename pattern section.

**-J, --job-name=<jobname>**
Specify a name for the job allocation. The specified name will appear along with
the job ID number when querying running jobs on the system. The default
is the name of the batch script, or just "sbatch" if the script is
read on sbatch's standard input.

**--kill-on-invalid-dep=<yes|no>**
If a job has an invalid dependency and it can never run this parameter tells
Slurm to terminate it or not. A terminated job state will be JOB_CANCELLED.
If this option is not specified the system wide behavior applies.
By default the job stays pending with reason DependencyNeverSatisfied or if the
kill_invalid_depend is specified in slurm.conf the job is terminated.

## Source excerpt: slurm_man_sbatch#39

**--mail-user=<user>**
User to receive email notification of state changes as defined by
--mail-type. This may be a full email address or a username. If a
username is specified, the value from MailDomain in slurm.conf will be
appended to create an email address.
The default value is the submitting user.

**--mcs-label=<mcs>**
Used only when a compatible MCSPlugin is enabled. This parameter is a
group that the user belongs to (mcs/group) or an arbitrary label string
(mcs/label). In both cases, no label will be assigned by default. Refer to
the MCS documentation for more details: <https://slurm.schedmd.com/mcs.html>

## Source excerpt: slurm_man_sbatch#55

**--oom-kill-step[={0|1}]**
Whether to kill the entire step if an OOM event is detected in any task of a
step. This overwrites the "OOMKillStep" setting in TaskPluginParam from
slurm.conf. When unset it will use the setting in slurm.conf. When set, a value
of "0" will disable killing the entire step, while a value of "1" will enable
it. This applies to the entire allocation except for the external step.
Default is "1" (enabled) when the option is found with no value.

**--open-mode={append|truncate}**
Open the output and error files using append or truncate mode as specified.
The default value is specified by the system configuration parameter
JobFileAppend.

## Source excerpt: slurm_man_sbatch#71

**--wait-all-nodes=<value>**
Controls when the execution of the command begins.
By default the job will begin execution as soon as the allocation is made.

**0**
Begin execution as soon as allocation can be made.
Do not wait for all nodes to be ready for use (i.e. booted).

**1**
Do not begin execution until all nodes are ready for use.

**--wckey=<wckey>**
Specify wckey to be used with job. If TrackWCKey=no (default) in the
slurm.conf this value is ignored.

**--wrap=<command_string>**
Sbatch will wrap the specified command string in a simple "sh" shell script,
and submit that script to the slurm controller. When --wrap is used,
a script name and arguments may not be specified on the command line; instead
the sbatch-generated wrapper script is used.

## Filename Pattern

sbatch allows for a filename pattern to contain one or more replacement
symbols, which are a percent sign "%" followed by a letter (e.g. %j).

**\\\\**
Do not process any of the replacement symbols.

**%%**
The character "%".

**%J**
jobid.stepid of the running job (e.g. "128.0"). The stepid is only expanded for
regular steps, not for special steps like "batch" or "extern".

**%j**
jobid of the running job.

**%N**
short hostname. This will create a separate IO file per node.

**%n**
Node identifier relative to current job (e.g. "0" is the first node of
the running job) This will create a separate IO file per node.

**%r**
Restart count of the running job.

**%S**
SLUID of the running job.

## Source excerpt: slurm_man_sbatch#72

**%s**
stepid of the running job.

**%t**
task identifier (rank) relative to current job. This will create a
separate IO file per task.

**%u**
User name.

**%x**
Job name.

A number placed between the percent character and format specifier may be
used to zero-pad the result in the IO filename to at minimum of specified
numbers. This number is ignored if the format specifier corresponds to
non-numeric data (%N for example). The maximal number is 10, if a value greater
than 10 is used the result is padding up to 10 characters.
Some examples of how the format string may be used for a 4 task job step with a
JobID of 128 and step id of 0 are included below:

**job%J.out**
job128.0.out

**job%4j.out**
job0128.out

**job%2j-%2t.out**
job128-00.out, job128-01.out, ...

## Performance

Executing sbatch sends a remote procedure call to slurmctld. If
enough calls from sbatch or other Slurm client commands that send remote
procedure calls to the slurmctld daemon come in at once, it can result in
a degradation of performance of the slurmctld daemon, possibly resulting
in a denial of service.

Do not run sbatch or other Slurm client commands that send remote
procedure calls to slurmctld from loops in shell scripts or other
programs. Ensure that programs limit calls to sbatch to the minimum
necessary for the information you are trying to gather.

## Input Environment Variables

## Source excerpt: upstream_sbatch_0b1faedc2d24

**--mail-type=<type>**
Notify user by email when certain event types occur.
Valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL (equivalent to
BEGIN, END, FAIL, INVALID_DEPEND, REQUEUE, and STAGE_OUT), INVALID_DEPEND
(dependency never satisfied), STAGE_OUT (burst buffer stage out and teardown
completed), TIME_LIMIT, TIME_LIMIT_90 (reached 90 percent of time limit),
TIME_LIMIT_80 (reached 80 percent of time limit), TIME_LIMIT_50 (reached 50
percent of time limit) and ARRAY_TASKS (send emails for each array task).
Multiple type values may be specified in a comma separated list.
NONE will suppress all event notifications, ignoring any other values specified.
By default no email notifications are sent.
The user to be notified is indicated with --mail-user.
Unless the ARRAY_TASKS option is specified, mail notifications on job BEGIN,
END, FAIL and REQUEUE apply to a job array as a whole rather than generating
individual email messages for each task in the job array.

## Source excerpt: upstream_sbatch_069aadf2d7e5

**-o, --output=<filename_pattern>**
Instruct Slurm to connect the batch script's standard output directly to the
file name specified in the "filename pattern".
By default both standard output and standard error are directed to the same file.
For job arrays, the default file name is "slurm-%A_%a.out", "%A" is replaced
by the job ID and "%a" with the array index.
For other jobs, the default file name is "slurm-%j.out", where the "%j" is
replaced by the job ID.
See the filename pattern section below for filename specification options.

## Source excerpt: upstream_sbatch_811dd5512820

**-e, --error=<filename_pattern>**
Instruct Slurm to connect the batch script's standard error directly to the
file name specified in the "filename pattern".
By default both standard output and standard error are directed to the same file.
For job arrays, the default file name is "slurm-%A_%a.out", "%A" is replaced
by the job ID and "%a" with the array index.
For other jobs, the default file name is "slurm-%j.out", where the "%j" is
replaced by the job ID.
See the filename pattern section below for filename specification options.

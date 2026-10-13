# Evaluation source bundle: debug

## Source excerpt: slurm_guide_faq#19

The most common problem is failing to set job time limits. If all jobs have the same time limit (for example the partition's time limit), then backfill will not be effective. Note that partitions can have both default and maximum time limits, which can be helpful in configuring a system for effective backfill scheduling.

In addition, there are a multitude of backfill scheduling parameters which can impact which jobs are considered for backfill scheduling, such as the maximum number of jobs tested per user. For more information see the slurm.conf man page and check the configuration of SchedulerParameters on your system.

#### Killed Jobs

Why is my job killed prematurely?

Slurm has a job purging mechanism to remove inactive jobs (resource allocations) before reaching its time limit, which could be infinite. This inactivity time limit is configurable by the system administrator. You can check its value with the command

scontrol show config | grep InactiveLimit

The value of InactiveLimit is in seconds. A zero value indicates that job purging is disabled. A job is considered inactive if it has no active job steps or if the srun command creating the job is not responding. In the case of a batch job, the srun command terminates after the job script is submitted. Therefore batch job pre- and post-processing is limited to the InactiveLimit. Contact your system administrator if you believe the InactiveLimit value should be changed.

## Source excerpt: slurm_guide_faq#20

Why is my batch job that launches no job steps being killed?

Slurm has a configuration parameter InactiveLimit intended to kill jobs that do not spawn any job steps for a configurable period of time. Your system administrator may modify the InactiveLimit to satisfy your needs. Alternately, you can just spawn a job step at the beginning of your script to execute in the background. It will be purged when your script exits or your job otherwise terminates. A line of this sort near the beginning of your script should suffice:

srun -N1 -n1 sleep 999999 &

What does "srun: Force Terminated job" indicate?

The srun command normally terminates when the standard output and error I/O from the spawned tasks end. This does not necessarily happen at the same time that a job step is terminated. For example, a file system problem could render a spawned task non-killable at the same time that I/O to srun is pending. Alternately a network problem could prevent the I/O from being transmitted to srun. In any event, the srun command is notified when a job step is terminated, either upon reaching its time limit or being explicitly killed. If the srun has not already terminated, the message "srun: Force Terminated job" is printed. If the job step's I/O does not terminate in a timely fashion thereafter, pending I/O is abandoned and the srun command exits.

What does this mean: "srun: First task exited 30s ago" followed by "srun Job Failed"?

## Source excerpt: slurm_guide_faq#21

The srun command monitors when tasks exit. By default, 30 seconds after the first task exits, the job is killed. This typically indicates some type of job failure and continuing to execute a parallel job when one of the tasks has exited is not normally productive. This behavior can be changed using srun's --wait=<time> option to either change the timeout period or disable the timeout altogether. See srun's man page for details.

#### Managing Jobs

How can I temporarily prevent a job from running (e.g. place it into a hold state)?

The easiest way to do this is to change a job's earliest begin time (optionally set at job submit time using the --begin option). The example below places a job into hold state (preventing its initiation for 30 days) and later permitting it to start now.

```

$ scontrol update JobId=1234 StartTime=now+30days
... later ...
$ scontrol update JobId=1234 StartTime=now

```

Can I change my job's size after it has started running?

Slurm supports the ability to decrease the size of jobs. Requesting fewer hardware resources, and changing partition, qos, reservation, licenses, etc. is only allowed for pending jobs.

## Source excerpt: slurm_guide_faq#25

When the srun command executes, it captures the resource limits in effect at submit time on the node where srun executes. These limits are propagated to the allocated nodes before initiating the user's job. The Slurm daemons running on the allocated nodes then try to establish identical resource limits for the job being initiated. There are several possible reasons for not being able to establish those resource limits.

- The hard resource limits applied to Slurm's slurmd daemon are lower than the user's soft resources limits on the submit host. Typically the slurmd daemon is initiated by the init daemon with the operating system default limits. This may be addressed either through use of the ulimit command in the /etc/sysconfig/slurm file or enabling PAM in Slurm.
- The user's hard resource limits on the allocated node are lower than the same user's soft hard resource limits on the node from which the job was submitted. It is recommended that the system administrator establish uniform hard resource limits for users on all nodes within a cluster to prevent this from occurring.
- PropagateResourceLimits or PropagateResourceLimitsExcept parameters are configured in slurm.conf and avoid propagation of specified limits.

## Source excerpt: slurm_guide_faq#26

NOTE: This may produce the error message "Can't propagate RLIMIT_...". The error message is printed only if the user explicitly specifies that the resource limit should be propagated or the srun command is running with verbose logging of actions from the slurmd daemon (e.g. "srun -d6 ...").

Why are jobs not getting the appropriate memory limit?

This is probably a variation on the locked memory limit problem described above. Use the same solution for the AS (Address Space), RSS (Resident Set Size), or other limits as needed.

Why is my MPI job failing due to the locked memory (memlock) limit being too low?

## Source excerpt: slurm_man_sacct#11

**-s, --state=<state_list>**
Selects jobs based on their state during the time period given.
Unless otherwise specified, the start and end time will be the
current time when the --state option is specified and
only currently running jobs can be displayed.
A start and/or end time must be specified to view information about
jobs not currently running.
See the JOB STATE CODES section below for a list of state designators.
Multiple state names may be specified using comma separators. Either the short
or long form of the state name may be used (e.g. CA or CANCELLED)
and the name is case insensitive (i.e. ca and CA both work).
NOTE: Note for a job to be selected in the PENDING state it must have
"EligibleTime" in the requested time interval or different from "Unknown". The
"EligibleTime" is displayed by the "scontrol show job" command. For example
jobs submitted with the "--hold" option will have "EligibleTime=Unknown" as
they are pending indefinitely.
NOTE: When specifying states and no start time is given the default
start time is 'now'. This is only when -j is not used. If -j is used the
start time will default to 'Epoch'. In both cases if no end time is given it
will default to 'now'. See the DEFAULT TIME WINDOW section (below) for
details about how this option changes the default values for --starttime
and --endtime.

**-K, --timelimit-max**
Ignored by itself, but if timelimit_min is set this will be the
maximum timelimit of the range. Default is no restriction.

## Source excerpt: slurm_man_sacct#23

**\f3TresUsageOutMaxTask**
Task for which each maximum TRES usage out occurred.

**TresUsageOutMin**
Tres minimum usage out by all tasks in job.

**TresUsageOutMinNode**
Node for which each minimum TRES usage out occurred.

**TresUsageOutMinTask**
Task for which each minimum TRES usage out occurred.

**\f3TresUsageOutTot**
Tres total usage out by all tasks in job.

**\f3UID**
The user identifier of the user who ran the job.

**\f3User**
The user name of the user who ran the job.

**\f3UserCPU**
The amount of user CPU time used by the job or job step. Format is the same as
\f3Elapsed.
NOTE: See the note for TotalCPU for information about how canceled jobs
are handled.

**\f3WCKey**
Workload  Characterization  Key.   Arbitrary  string for grouping orthogonal accounts together.

**\f3WCKeyID**
Reference to the wckey.

**WorkDir**
The directory used by the job to execute commands.

## Job State Codes

The following states are recognized by sacct. A full list of possible states
is available at <https://slurm.schedmd.com/job_state_codes.html>.

**\f3CA  CANCELLED**
Job was explicitly cancelled by the user or system administrator.
The job may or may not have been initiated.

**\f3CD  COMPLETED**
Job has terminated all processes on all nodes with an exit code of zero.

**\f3DL  DEADLINE**
Job terminated on deadline.

**\f3F   FAILED**
Job terminated with non-zero exit code or other failure condition.

**\f3NF  NODE_FAIL**
Job terminated due to failure of one or more allocated nodes.

## Source excerpt: slurm_man_sacct#24

**\f3OOM OUT_OF_MEMORY**
Job experienced out of memory error.

**\f3PD  PENDING**
Job is awaiting resource allocation.

**\f3R   RUNNING**
Job currently has an allocation.

**\f3RS  RESIZING**
Job is about to change size.

**\f3RV  REVOKED**
Sibling was removed from cluster due to other cluster starting the job.

**\f3S   SUSPENDED**
Job has an allocation, but execution has been suspended and CPUs have been
released for other jobs.

**\f3TO  TIMEOUT**
Job terminated upon reaching its time limit.

## Default Time Window

The options --starttime and --endtime define the time window between
which sacct is going to search. For historical and practical
reasons their default values (i.e. the default time window)
depends on other options: --jobs and --state.

Depending on if --jobs and/or --state are specified, the default
values of --starttime  and --endtime options are:

WITHOUT EITHER --jobs NOR --state specified:

--starttime defaults to Midnight.

--endtime defaults to Now.

WITH --jobs AND WITHOUT --state specified:

--starttime defaults to Epoch 0.

--endtime defaults to Now.

WITHOUT --jobs AND WITH --state specified:

--starttime defaults to Now.

--endtime defaults to --starttime and to Now if --starttime is not specified.

WITH BOTH --jobs AND --state specified:

--starttime defaults to Epoch 0.

--endtime defaults to --starttime or to Now if --starttime is not specified.

NOTE: With -v/--verbose a message about the actual time
window in use is shown.

## Performance

## Source excerpt: slurm_guide_job_exit_code#0

## Job Exit Codes

A job's exit code (aka exit status, return code and completion code) is captured by Slurm and saved as part of the job record. For sbatch jobs, the exit code that is captured is the output of the batch script. For salloc jobs, the exit code will be the return value of the exit call that terminates the salloc session. For srun, the exit code will be the return value of the command that srun executes.

Any non-zero exit code will be assumed to be a job failure and will result in a Job State of FAILED with a Reason of "NonZeroExitCode".

The exit code is an 8 bit unsigned number ranging between 0 and 255. While it is possible for a job to return a negative exit code, Slurm will display it as an unsigned value in the 0 - 255 range.

### Job Step Exit Codes

When a job contains multiple job steps, the exit code of each executable invoked by srun is saved individually to the job step record.

### Signaled Jobs

Slurm displays a job's exit code in the output of the scontrol show job and the sview utility. Slurm displays job step exit codes in the output of the scontrol show step and the sview utility.

### Database Job/Step Records

The Slurm control daemon sends job and step records to the Slurm database when the Slurm accounting_storage plugin is installed. Job and step records sent to the Slurm db can be viewed using the sacct command. The default sacct output contains an ExitCode field whose format mirrors the output of scontrol and sview described above.

## Source excerpt: slurm_guide_job_exit_code#1

## Derived Exit Code and Comment String

After reading the above description of a job's exit code, one can imagine a scenario where a central task of a batch job fails but the script returns an exit code of zero, indicating success. In many cases, a user may not be able to ascertain the success or failure of a job until after they have examined the job's output files.

The job includes a "derived exit code" field. It is initially set to the value of the highest exit code returned by all of the job's steps (srun invocations). The job's derived exit code is determined by the Slurm control daemon and sent to the database when the accounting_storage plugin is enabled.

In addition to the derived exit code, the job record in the Slurm database contains a comment string. This is initialized to the job's comment string (when AccountingStoreFlags parameter in the slurm.conf contains 'job_comment') and can only be changed by the user.

A new option has been added to the sacctmgr command to provide the user the means to modify these two fields of the job record. No other modification to the job record is allowed. For those who prefer a simpler command specifically designed to view and modify the derived exit code and comment string, the sjobexitmod wrapper has been created (see below).

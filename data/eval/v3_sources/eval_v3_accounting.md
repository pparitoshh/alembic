# Evaluation source bundle: accounting

## Source excerpt: slurm_man_sacct#0

## Name

sacct - displays accounting data for all jobs and job steps in the
Slurm job accounting log or Slurm database

## Synopsis

sacct [OPTIONS...]

## Description

Accounting information for jobs invoked with Slurm are either logged
in the job accounting log file or saved to the Slurm database, as
configured with the AccountingStorageType parameter.

The \f3sacct command displays job accounting data stored in the job
accounting log file or Slurm database in a variety of forms for your
analysis. The \f3sacct command displays information on jobs, job
steps, status, and exitcodes by default. You can tailor the output
with the use of the \f3--format= option to specify the fields to
be shown.

Job records consist of a primary entry for the job as a whole as well as
entries for job steps. The Job Launch page has a more detailed description
of each type of job step.
<https://slurm.schedmd.com/job_launch.html#job_record>

For the root user, the \f3sacct command displays job accounting
data for all users, although there are options to filter the output to
report only the jobs from a specified user or group.

For the non-root user, the \f3sacct command limits the display of
job accounting data to jobs that were launched with their own user
identifier (UID) by default. Data for other users can be displayed
with the \f3--allusers, \f3--user, or \f3--uid options.

## Source excerpt: slurm_man_sacct#2

**-a, --allusers**
Displays all users' jobs when run by user root or if PrivateData is not
configured to jobs.
Otherwise display the current user's jobs

**-x, --associations=<assoc_list>**
Displays the statistics only for the jobs running under the
association ids specified by the assoc_list operand, which is a
comma-separated list of association ids. Space characters are not
allowed in the assoc_list. Default is all associations.

**-B, --batch-script**
This option will print the batch script of job if the job used one. If the job
didn't have a script 'NONE' is output.
NOTE: AccountingStoreFlags=job_script is required for this.
NOTE: Requesting specific job(s) with '-j' is required for this.

**-b, --brief**
Displays a brief listing consisting of JobID, State, and ExitCode.

**-M, --clusters=<cluster_list>**
Displays the statistics only for the jobs started on the clusters
specified by the cluster_list operand, which is a
comma-separated list of clusters. Space characters are not allowed
in the cluster_list.
A value of 'all' will query to run on all clusters.
The default is current cluster you are executing the sacct command on or
all clusters in the federation when executed on a federated cluster.
This option implicitly sets the --local option.

**-c, --completion**
Use job completion data instead of job accounting. The JobCompType
parameter in the slurm.conf file must be defined to a non-none option.
Does not support federated cluster information (local data only).

## Source excerpt: slurm_man_sacct#5

**-e, --helpformat**
Print a list of fields that can be specified with the --format option.

## Source excerpt: slurm_man_sacct#6

```
.ft 3
Fields available:

Account             AdminComment        AllocCPUS           AllocNodes
AllocTRES           AssocID             AveCPU              AveCPUFreq
AveDiskRead         AveDiskWrite        AvePages            AveRSS
AveVMSize           BlockID             Cluster             Comment
Constraints         ConsumedEnergy      ConsumedEnergyRaw   Container
CPUTime             CPUTimeRAW          DBIndex             DerivedExitCode
Elapsed             ElapsedRaw          Eligible            End
Exclusive           ExitCode            Extra               FailedNode
Flags               GID                 Group               JobID
JobIDRaw            JobName             Layout              Licenses
MaxDiskRead         MaxDiskReadNode     MaxDiskReadTask     MaxDiskWrite
MaxDiskWriteNode    MaxDiskWriteTask    MaxPages            MaxPagesNode
MaxPagesTask        MaxRSS              MaxRSSNode          MaxRSSTask
MaxVMSize           MaxVMSizeNode       MaxVMSizeTask       McsLabel
MinCPU              MinCPUNode          MinCPUTask          NCPUS
NNodes              NodeList            NTasks              OriginalSLUID
OverSubscribe       Partition           Planned             PlannedCPU
PlannedCPURAW       Priority            QOS                 QOSRAW
QOSREQ              Reason              ReqCPUFreq          ReqCPUFreqGov
ReqCPUFreqMax       ReqCPUFreqMin       ReqCPUS             ReqMem
ReqNodes            ReqReservation      ReqTRES             Reservation
ReservationId       Restarts            SegmentSize         SLUID
Start               State               StdErr              StdIn
StdOut              Submit              SubmitLine          Suspended
SystemComment       SystemCPU           Timelimit           TimelimitRaw
TotalCPU            TRESUsageInAve      TRESUsageInMax      TRESUsageInMaxNode
TRESUsageInMaxTask  TRESUsageInMin      TRESUsageInMinNode  TRESUsageInMinTask
TRESUsageInTot      TRESUsageOutAve     TRESUsageOutMax     TRESUsageOutMaxNode
TRESUsageOutMaxTask TRESUsageOutMin     TRESUsageOutMinNode TRESUsageOutMinTask
TRESUsageOutTot     UID                 User                UserCPU
WCKey               WCKeyID             WorkDir
.ft 1
```

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

## Source excerpt: slurm_man_sacct#27

This example shows the same job accounting information with the
\f3brief option.

```
.ft 3
# sacct --brief
     Jobid     State  ExitCode
---------- ---------- --------
2          RUNNING           0
3          RUNNING           0
4          RUNNING           0
4.0        COMPLETED         0
.ft 1
```

```
.ft 3
# sacct --allocations
Jobid      Jobname    Partition Account    AllocCPUS  State     ExitCode
---------- ---------- ---------- ---------- ------- ---------- --------
3          sja_init   andy       acct1            1 COMPLETED         0
4          sjaload    andy       acct1            2 COMPLETED         0
5          sja_scr1   andy       acct1            1 COMPLETED         0
6          sja_scr2   andy       acct1           18 COMPLETED         2
7          sja_scr3   andy       acct1           18 COMPLETED         0
8          sja_scr5   andy       acct1            2 COMPLETED         0
9          sja_scr7   andy       acct1           90 COMPLETED         1
10         endscript  andy       acct1          186 COMPLETED         0

.ft 1
```

This example demonstrates the ability to customize the output of the
\f3sacct command. The fields are displayed in the order designated
on the command line.

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

## Source excerpt: slurm_guide_job_exit_code#2

The user now has the means to annotate a job's exit code after it completes and provide a description of what failed. This includes the ability to annotate a successful completion to jobs that appear to have failed but actually succeeded.

### The sjobexitmod command

The sjobexitmod command is available to display and update the two derived exit fields of the Slurm db's job record. sjobexitmod can first be used to display the existing exit code / string for a job:

```

> sjobexitmod -l 123
JobID Account NNodes NodeList     State ExitCode DerivedExitCode Comment
----- ------- ------ -------- --------- -------- --------------- -------
123        lc      1     tux0 COMPLETED      0:0             0:0

```

If a change is desired, sjobexitmod can modify the derived fields:

```

> sjobexitmod -e 49 -r "out of memory" 123

 Modification of job 123 was successful.

> sjobexitmod -l 123
JobID Account NNodes NodeList     State ExitCode DerivedExitCode Comment
----- ------- ------ -------- --------- -------- --------------- -------
123        lc      1     tux0 COMPLETED      0:0            49:0 out of memory

```

The existing sacct command also supports the two new derived exit fields:

```

> sacct -X -j 123 -o JobID,NNodes,State,ExitCode,DerivedExitcode,Comment
JobID   NNodes      State ExitCode DerivedExitCode        Comment
------ ------- ---------- -------- --------------- --------------
123          1  COMPLETED      0:0            49:0  out of memory

```

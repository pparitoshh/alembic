# Evaluation source bundle: dependencies

## Source excerpt: slurm_man_sbatch#0

## Name

sbatch - Submit a batch script to Slurm.

## Synopsis

sbatch [OPTIONS(0)...] [ : [OPTIONS(N)...]] script(0) [args(0)...]

Option(s) define multiple jobs in a co-scheduled heterogeneous job.
For more details about heterogeneous jobs see the document

https://slurm.schedmd.com/heterogeneous_jobs.html

## Description

sbatch submits a batch script to Slurm. The batch script may be given to
sbatch through a file name on the command line, or if no file name is specified,
sbatch will read in a script from standard input.

The batch script may contain one or more lines beginning with "#SBATCH" followed
by any of the CLI options documented on this page. #SBATCH directives are read
directly by Slurm, so shell-specific syntax including variable names will be
read as literal text. Once the first non-comment, non-whitespace line has been
reached in the script, no more #SBATCH directives will be processed. See example
below.

sbatch exits immediately after the script is successfully transferred to the
Slurm controller and assigned a Slurm job ID. The batch script is not
necessarily granted resources immediately, it may sit in the queue of pending
jobs for some time before its required resources become available.

## Source excerpt: slurm_man_sbatch#17

**--delay-boot=<minutes>**
Do not reboot nodes in order to satisfied this job's feature specification if
the job has been eligible to run for less than this time period.
If the job has waited for less than the specified period, it will use only
nodes which already have the specified features.
The argument is in units of minutes.
A default value may be set by a system administrator using the delay_boot
option of the SchedulerParameters configuration parameter in the
slurm.conf file, otherwise the default value is zero (no delay).

**-d, --dependency=<dependency_list>**
Defer the start of this job until the specified dependencies have been
satisfied. Once a dependency is satisfied, it is removed from the job.
<dependency_list> is of the form
<type:job_id[:job_id][,type:job_id[:job_id]]> or
<type:job_id[:job_id][?type:job_id[:job_id]]>.
All dependencies must be satisfied if the "," separator is used.
Any dependency may be satisfied if the "?" separator is used.
Only one separator may be used. For instance:

```
-d afterok:20:21,afterany:23
```

means that the job can run only after a 0 return code of jobs 20 and 21
AND the completion of job 23. However:

```
-d afterok:20:21?afterany:23
```

**after:job_id[[+time][:jobid[+time]...]]**
After the specified jobs start or are cancelled and 'time' in minutes from job
start or cancellation happens, this
job can begin execution. If no 'time' is given then there is no delay after
start or cancellation.

## Source excerpt: slurm_man_sbatch#18

**afterany:job_id[:jobid...]**
This job can begin execution after the specified jobs have terminated.
This is the default dependency type.

**afterburstbuffer:job_id[:jobid...]**
This job can begin execution after the specified jobs have terminated and
any associated burst buffer stage out operations have completed.

**afternotok:job_id[:jobid...]**
This job can begin execution after the specified jobs have terminated
in some failed state (non-zero exit code, node failure, timed out, etc).
This job must be submitted while the specified job is still active or within
MinJobAge seconds after the specified job has ended.
If the dependent job ID is not found and is on the same cluster as the job
submission, the job is rejected. If the dependent job ID is not found and is on
a different cluster from the job submission, the dependency is marked as
failed.

**afterok:job_id[:jobid...]**
This job can begin execution after the specified jobs have successfully
executed (ran to completion with an exit code of zero).
This job must be submitted while the specified job is still active or within
MinJobAge seconds after the specified job has ended.
If the dependent job ID is not found and is on the same cluster as the job
submission, the job is rejected. If the dependent job ID is not found and is on
a different cluster from the job submission, the dependency is marked as
failed.

## Source excerpt: slurm_man_sbatch#19

**singleton**
This job can begin execution after any previously launched jobs
sharing the same job name and user have terminated.
In other words, only one job by that name and owned by that user can be running
or suspended at any point in time.
In a federation, a singleton dependency must be fulfilled on all clusters
unless DependencyParameters=disable_remote_singleton is used in slurm.conf.

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

## Source excerpt: slurm_man_sbatch#57

**-s, --oversubscribe**
The job allocation can over-subscribe resources with other running jobs.
The resources to be over-subscribed can be nodes, sockets, cores, and/or
hyperthreads depending upon configuration.
The default over-subscribe behavior depends on system configuration and the
partition's OverSubscribe option takes precedence over the job's option.
This option may result in the allocation being granted sooner than if the
--oversubscribe option was not set and allow higher system utilization, but
application performance will likely suffer due to competition for resources.
Also see the --exclusive option.
NOTE: This option is mutually exclusive with --exclusive.

**--parsable**
Outputs only the job ID number and the cluster name if present.
The values are separated by a semicolon. Errors will still be displayed.
Note that --quiet suppresses the job ID along with the other
informational messages.

# Day 0 — Leonardo Setup for DistillKit

This guide is for a teammate using **CINECA Leonardo for the first time**.

You do **not** need to download all models or install the heavy DistillKit environment again. Those are already available in our shared project storage.

This guide only covers what **each teammate needs to set up personally**:

1. Log in to Leonardo with username + password
2. Configure `ssh leonardo`
3. Configure Git
4. Access the DistillKit project
5. Connect Leonardo to GitHub
6. Create/use your personal Git clone
7. Switch to the correct branch
8. Verify the shared DistillKit environment

---

# 0. Important project information

Our Leonardo project account:

```text
EUHPC_D30_031
```

Shared DistillKit runtime repository:

```text
/leonardo_work/EUHPC_D30_031/alembic/alembic
```

GitHub repository:

```text
git@github.com:pparitoshh/alembic.git
```

Your Leonardo username will look something like:

```text
a08trcXX
```

For example:

```text
a08trc09
```

Always replace `a08trcXX` with **your own username**.

---

# 1. First login to Leonardo

## Run this on YOUR LAPTOP

Open:

- **macOS/Linux:** Terminal
- **Windows:** PowerShell

Then run:

```bash
ssh a08trcXX@login01-ext.leonardo.cineca.it
```

Example:

```bash
ssh a08trc09@login01-ext.leonardo.cineca.it
```

The first time, you may see:

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type:

```text
yes
```

and press Enter.

Then Leonardo asks for your password:

```text
a08trcXX@login01-ext.leonardo.cineca.it's password:
```

Enter the password given by the hackathon organisers.

> While typing an SSH password, **nothing appears on the screen**.  
> No dots, stars, or characters are shown.  
> This is completely normal.

If login succeeds, you should see something similar to:

```text
[a08trcXX@login01 ~]$
```

You are now inside Leonardo.

---

# 2. Create an easier `ssh leonardo` login

Typing this every time is annoying:

```bash
ssh a08trcXX@login01-ext.leonardo.cineca.it
```

We will configure your laptop so you can simply use:

```bash
ssh leonardo
```

---

## 2.1 Exit Leonardo

If you are currently inside Leonardo, run:

```bash
exit
```

You should now be back on **your laptop**.

---

## 2.2 Create an SSH key on YOUR LAPTOP

Run:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/leonardo -C "leonardo"
```

You may see:

```text
Enter passphrase (empty for no passphrase):
```

If you do not want to use a passphrase, press:

```text
Enter
Enter
```

This creates:

```text
~/.ssh/leonardo
~/.ssh/leonardo.pub
```

Important:

```text
~/.ssh/leonardo       = PRIVATE KEY
~/.ssh/leonardo.pub   = PUBLIC KEY
```

### Never share:

```text
~/.ssh/leonardo
```

### The `.pub` file is safe to copy:

```text
~/.ssh/leonardo.pub
```

---

# 3. Add your laptop's public key to Leonardo

## Run this on YOUR LAPTOP

```bash
cat ~/.ssh/leonardo.pub | \
ssh a08trcXX@login01-ext.leonardo.cineca.it \
'umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys'
```

Replace:

```text
a08trcXX
```

with your own Leonardo username.

It will ask for your Leonardo password one more time.

Enter the password given by the organisers.

After this, your laptop's SSH key is registered on Leonardo.

---

# 4. Configure the `ssh leonardo` shortcut

## Run this on YOUR LAPTOP

Open:

```bash
nano ~/.ssh/config
```

If `nano` is not available, open `~/.ssh/config` using any text editor.

Add:

```text
Host leonardo
    HostName login01-ext.leonardo.cineca.it
    User a08trcXX
    SetEnv LANG=C
    IdentityFile ~/.ssh/leonardo
    IdentitiesOnly yes
```

Replace:

```text
a08trcXX
```

with your own username.

Example:

```text
Host leonardo
    HostName login01-ext.leonardo.cineca.it
    User a08trc09
    SetEnv LANG=C
    IdentityFile ~/.ssh/leonardo
    IdentitiesOnly yes
```

If you are using nano, save with:

```text
Ctrl + O
Enter
Ctrl + X
```

Then run:

```bash
chmod 600 ~/.ssh/config
```

---

# 5. Test the shortcut

## Run this on YOUR LAPTOP

```bash
ssh leonardo
```

If everything is correct, you should reach:

```text
[a08trcXX@login01 ~]$
```

From now on, whenever you want to connect to Leonardo, simply use:

```bash
ssh leonardo
```

Optional test:

```bash
ssh -o BatchMode=yes leonardo 'echo "SSH KEY WORKS"'
```

Expected:

```text
SSH KEY WORKS
```

---

# 6. Configure Git on Leonardo

From this point onward, commands marked **ON LEONARDO** should be run after:

```bash
ssh leonardo
```

Check whether your Git identity already exists:

```bash
git config --global user.name
```

and:

```bash
git config --global user.email
```

If nothing appears, configure them:

```bash
git config --global user.name "Your Name"
```

```bash
git config --global user.email "your-github-email@example.com"
```

Example:

```bash
git config --global user.name "VaibhavKKM"
git config --global user.email "your-email@example.com"
```

Verify:

```bash
git config --global --list
```

You should see:

```text
user.name=Your Name
user.email=your-github-email@example.com
```

---

# 7. Go to the shared DistillKit project

## ON LEONARDO

Run:

```bash
cd /leonardo_work/EUHPC_D30_031/alembic/alembic
```

Check:

```bash
pwd
```

Expected:

```text
/leonardo_work/EUHPC_D30_031/alembic/alembic
```

Then:

```bash
git status
```

---

# 8. If Git reports "dubious ownership"

Because this is a shared repository, some files were created by another teammate.

Git may show:

```text
fatal: detected dubious ownership in repository
```

If this happens, run:

```bash
git config --global --add safe.directory \
/leonardo_work/EUHPC_D30_031/alembic/alembic
```

Then retry:

```bash
git status
```

---

# 9. Set the DistillKit variables

## ON LEONARDO

Run:

```bash
export SBATCH_ACCOUNT=EUHPC_D30_031
```

Then:

```bash
export REPO="/leonardo_work/EUHPC_D30_031/alembic/alembic"
```

Check:

```bash
echo "$SBATCH_ACCOUNT"
```

Expected:

```text
EUHPC_D30_031
```

Check:

```bash
echo "$REPO"
```

Expected:

```text
/leonardo_work/EUHPC_D30_031/alembic/alembic
```

---

# 10. Optional: automatically set the variables after every login

Without this step, the variables disappear when you log out of Leonardo.

To save them permanently for your Leonardo account:

```bash
cat >> ~/.bashrc <<'EOF'

# DistillKit on Leonardo
export SBATCH_ACCOUNT=EUHPC_D30_031
export REPO="/leonardo_work/EUHPC_D30_031/alembic/alembic"
EOF
```

Then:

```bash
source ~/.bashrc
```

Check:

```bash
echo "$SBATCH_ACCOUNT"
echo "$REPO"
```

---

# 11. IMPORTANT — do NOT reinstall the heavy setup

The heavy DistillKit environment has already been created in our **shared `$WORK` storage**.

New teammates should **not download everything again**.

Do NOT:

```text
- download the teacher model again
- download the student model again
- download the judge models again
- create another shared vLLM environment
- rebuild the shared llama.cpp setup
- create a new Pixi environment
- rerun the complete heavy setup script
```

In particular, do **not** run:

```bash
bash slurm/setup_login.sh
```

unless the team explicitly asks you to.

The required environment and models are already shared.

---

# 12. Shared software locations

The shared DistillKit Python environment is:

```text
$REPO/.venv
```

The shared vLLM environment is:

```text
$WORK/venvs/vllm
```

The shared Hugging Face cache is:

```text
$WORK/hf_cache
```

The Leonardo-compatible llama.cpp installation is:

```text
$WORK/tools/llama.cpp
```

Check them:

```bash
ls -ld \
"$REPO/.venv" \
"$WORK/venvs/vllm" \
"$WORK/hf_cache" \
"$WORK/tools/llama.cpp"
```

---

# 13. Models already downloaded

These models are already available in the shared Hugging Face cache.

## Teacher

```text
Qwen/Qwen3-32B-AWQ
```

## Student

```text
Qwen/Qwen3-4B-Instruct-2507
```

## Main judge

```text
openai/gpt-oss-20b
```

## Cross-check judge

```text
google/gemma-4-26B-A4B-it
```

Do **not** download them separately again.

---

# 14. Test the shared DistillKit environment

## ON LEONARDO

Go to the shared runtime repository:

```bash
cd /leonardo_work/EUHPC_D30_031/alembic/alembic
```

Set the variables:

```bash
export SBATCH_ACCOUNT=EUHPC_D30_031
export REPO="$PWD"
```

Load the project settings:

```bash
source slurm/env.sh
```

Run the tests:

```bash
"$VENV/bin/python" -m pytest -q
```

A healthy setup should finish with all tests passing.

At the time this guide was written, the expected result was:

```text
98 passed
```

---

# 15. Connect Leonardo to your GitHub account

This is different from the earlier SSH setup.

Earlier we configured:

```text
YOUR LAPTOP → LEONARDO
```

Now we configure:

```text
LEONARDO → GITHUB
```

This is required if you want to:

```text
git clone
git pull
git push
```

using GitHub SSH.

---

# 16. First test GitHub SSH

## ON LEONARDO

Run:

```bash
ssh -T git@github.com
```

The first time, you may see:

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type:

```text
yes
```

and press Enter.

If you then see:

```text
Permission denied (publickey).
```

that is okay.

It simply means Leonardo does not yet have a key connected to your GitHub account.

Continue with the next step.

---

# 17. Create a GitHub SSH key ON LEONARDO

Run:

```bash
ssh-keygen -t ed25519 \
  -f ~/.ssh/github_leonardo \
  -C "your-github-email@example.com"
```

Replace the email with the email associated with your GitHub account.

When asked for a passphrase, either create one or press:

```text
Enter
Enter
```

This creates:

```text
~/.ssh/github_leonardo
~/.ssh/github_leonardo.pub
```

Important:

```text
github_leonardo       = PRIVATE KEY — NEVER SHARE
github_leonardo.pub   = PUBLIC KEY — safe to copy
```

---

# 18. Copy your GitHub public key

## ON LEONARDO

Run:

```bash
cat ~/.ssh/github_leonardo.pub
```

You will see one long line beginning with:

```text
ssh-ed25519
```

Copy the **entire line**.

---

# 19. Add the key to GitHub

Open GitHub in your browser.

Go to:

```text
Profile picture
→ Settings
→ SSH and GPG keys
→ New SSH key
```

Use:

```text
Title: Leonardo - a08trcXX
Key type: Authentication Key
```

Replace `a08trcXX` with your own Leonardo username.

Paste the complete public key.

Save it.

---

# 20. Tell Leonardo which key to use for GitHub

## ON LEONARDO

Open:

```bash
nano ~/.ssh/config
```

Your Leonardo `~/.ssh/config` should contain a GitHub section like:

```text
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_leonardo
    IdentitiesOnly yes
```

Save:

```text
Ctrl + O
Enter
Ctrl + X
```

Then:

```bash
chmod 600 ~/.ssh/config
```

---

# 21. Test GitHub again

## ON LEONARDO

Run:

```bash
ssh -T git@github.com
```

The first time, GitHub may again ask:

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type:

```text
yes
```

If everything is correct, you should see something similar to:

```text
Hi YOUR_GITHUB_USERNAME! You've successfully authenticated, but GitHub does not provide shell access.
```

For example:

```text
Hi Vaibhavkkm! You've successfully authenticated, but GitHub does not provide shell access.
```

This means **everything is working correctly**.

> `GitHub does not provide shell access` is normal.  
> It is **not an error**.

---

# 22. Use a personal Git clone for code/document changes

The repository under:

```text
/leonardo_work/EUHPC_D30_031/alembic/alembic
```

is a **shared runtime repository**.

Some files there belong to other teammates.

That can cause permission problems when trying to edit or commit files.

Therefore:

### Use the shared `$WORK` repo for:

```text
- shared environments
- models
- SLURM jobs
- shared data
- shared runs
```

### Use your own `$HOME` clone for:

```text
- editing code
- writing documentation
- Git commits
- Git pushes
- branches
```

---

# 23. Clone the GitHub repository into your personal `$HOME`

## ON LEONARDO

Go home:

```bash
cd ~
```

Clone:

```bash
git clone git@github.com:pparitoshh/alembic.git alembic-dev
```

Enter the clone:

```bash
cd ~/alembic-dev
```

Check:

```bash
git status
```

Then:

```bash
git remote -v
```

You should see the GitHub repository as `origin`.

---

# 24. Switch to the `DAY_0_PG` branch

The branch already exists on GitHub.

Inside:

```text
~/alembic-dev
```

run:

```bash
git fetch origin --prune
```

Check:

```bash
git branch -a | grep DAY_0_PG
```

You should see something similar to:

```text
remotes/origin/DAY_0_PG
```

Then run:

```bash
git switch --track origin/DAY_0_PG
```

If Git tells you that `DAY_0_PG` already exists locally, simply use:

```bash
git switch DAY_0_PG
```

Verify:

```bash
git branch --show-current
```

Expected:

```text
DAY_0_PG
```

Pull the latest changes:

```bash
git pull --ff-only origin DAY_0_PG
```

---

# 25. Create/edit a documentation file

For example:

```bash
cd ~/alembic-dev
```

Then:

```bash
vim DAY_0_LEONARDO_SETUP.md
```

or:

```bash
nano DAY_0_LEONARDO_SETUP.md
```

Paste the Markdown content and save it.

---

# 26. Check your Git changes

Run:

```bash
git status
```

You should see your new or modified file.

For example:

```text
Untracked files:
    DAY_0_LEONARDO_SETUP.md
```

---

# 27. Add the file

Run:

```bash
git add DAY_0_LEONARDO_SETUP.md
```

Check:

```bash
git status
```

---

# 28. Commit the file

Run:

```bash
git commit -m "docs: add Leonardo day 0 setup guide"
```

---

# 29. Push the branch to GitHub

Run:

```bash
git push -u origin DAY_0_PG
```

After the first successful push, future pushes can normally use:

```bash
git push
```

---

# 30. Understanding the Leonardo login node

When you connect using:

```bash
ssh leonardo
```

you normally arrive on a **login node**, for example:

```text
login01
```

Use the login node for:

```text
- SSH
- Git
- editing files
- preparing job scripts
- light setup
- checking files
- submitting SLURM jobs
```

Do **not** run heavy GPU workloads directly on the login node.

---

# 31. GPU work runs through SLURM

Leonardo GPU jobs are submitted using:

```bash
sbatch ...
```

Check your current jobs with:

```bash
squeue --me
```

Common job states:

```text
PD = Pending / waiting
R  = Running
```

If nothing appears, you currently have no active jobs.

---

# 32. Check previous jobs

Run:

```bash
sacct -u "$USER" -S today \
--format=JobID,JobName,State,ExitCode,Elapsed
```

Common states include:

```text
COMPLETED
FAILED
CANCELLED
TIMEOUT
```

---

# 33. Shared directory permissions

Because this is a shared project, some directories may have been created by another teammate.

Check the shared log directory:

```bash
cd /leonardo_work/EUHPC_D30_031/alembic/alembic
```

Then:

```bash
test -w slurm/logs \
&& echo "LOGS WRITABLE" \
|| echo "LOGS NOT WRITABLE"
```

Check the shared runs directory:

```bash
test -w runs \
&& echo "RUNS WRITABLE" \
|| echo "RUNS NOT WRITABLE"
```

Ideally you should see:

```text
LOGS WRITABLE
RUNS WRITABLE
```

If not, contact the owner/team lead.

Do **not** randomly change another user's file permissions.

---

# 34. Useful everyday commands

## Connect to Leonardo

### ON YOUR LAPTOP

```bash
ssh leonardo
```

---

## Go to the shared runtime project

### ON LEONARDO

```bash
cd /leonardo_work/EUHPC_D30_031/alembic/alembic
```

---

## Set project variables

```bash
export SBATCH_ACCOUNT=EUHPC_D30_031
export REPO="/leonardo_work/EUHPC_D30_031/alembic/alembic"
```

---

## Check running jobs

```bash
squeue --me
```

---

## Go to your personal Git clone

```bash
cd ~/alembic-dev
```

---

## Check your current branch

```bash
git branch --show-current
```

---

## Get the newest changes

```bash
git pull
```

---

## Check your changes

```bash
git status
```

---

# 35. Common problems

## Problem: password does not appear while typing

This is normal.

SSH hides password input completely.

Type the password and press Enter.

---

## Problem: first SSH connection asks this

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type:

```text
yes
```

and press Enter.

---

## Problem: `fatal: detected dubious ownership`

Run:

```bash
git config --global --add safe.directory \
/leonardo_work/EUHPC_D30_031/alembic/alembic
```

---

## Problem: `$REPO` prints nothing

For example:

```bash
echo "$REPO"
```

returns an empty line.

Set it again:

```bash
export REPO="/leonardo_work/EUHPC_D30_031/alembic/alembic"
```

Also set:

```bash
export SBATCH_ACCOUNT=EUHPC_D30_031
```

---

## Problem: `Permission denied (publickey)` for GitHub

Test:

```bash
ssh -T git@github.com
```

If you see:

```text
Permission denied (publickey).
```

create/add the `github_leonardo` SSH key using the steps above.

---

## Problem: `git status` says "not a git repository"

Example:

```text
fatal: not a git repository
```

You are probably in the wrong directory.

For your personal clone:

```bash
cd ~/alembic-dev
```

For the shared runtime repo:

```bash
cd /leonardo_work/EUHPC_D30_031/alembic/alembic
```

Then run:

```bash
git status
```

---

## Problem: Git clone failed

If this failed:

```bash
git clone git@github.com:pparitoshh/alembic.git alembic-dev
```

first check:

```bash
ssh -T git@github.com
```

Only retry the clone once GitHub says:

```text
You've successfully authenticated
```

Then:

```bash
cd ~
rm -rf ~/alembic-dev
git clone git@github.com:pparitoshh/alembic.git alembic-dev
```

---

## Problem: `pixi: command not found`

Do **not** install Pixi for the shared DistillKit environment.

The project already has its required shared environment.

---

## Problem: this path does not exist

```bash
cd $WORK/your_team_name
```

`your_team_name` is only a placeholder from the generic Leonardo documentation.

Our actual DistillKit path is:

```bash
cd /leonardo_work/EUHPC_D30_031/alembic/alembic
```

---

## Problem: permission denied inside `slurm/logs`

This means the shared directory is not writable by your account.

Contact the team member who owns the directory.

Do not try to modify files owned by another user unless agreed by the team.

---

# 36. SSH security rules

## Safe to share

Files ending in:

```text
.pub
```

For example:

```text
~/.ssh/leonardo.pub
~/.ssh/github_leonardo.pub
```

---

## NEVER share

```text
~/.ssh/leonardo
~/.ssh/github_leonardo
```

Never paste private SSH keys into:

```text
GitHub
Slack
Discord
email
ChatGPT
documents
```

---

# 37. Very short setup summary

## First-ever login

### ON YOUR LAPTOP

```bash
ssh a08trcXX@login01-ext.leonardo.cineca.it
```

Type:

```text
yes
```

when asked.

Then enter the password provided by the organisers.

---

## After your laptop SSH setup

Use:

```bash
ssh leonardo
```

---

## Once inside Leonardo

```bash
export SBATCH_ACCOUNT=EUHPC_D30_031
export REPO="/leonardo_work/EUHPC_D30_031/alembic/alembic"

cd "$REPO"
```

---

## Test DistillKit

```bash
source slurm/env.sh
"$VENV/bin/python" -m pytest -q
```

---

## Test GitHub SSH

```bash
ssh -T git@github.com
```

You want:

```text
Hi YOUR_GITHUB_USERNAME! You've successfully authenticated, but GitHub does not provide shell access.
```

---

## Personal Git workspace

```bash
cd ~
git clone git@github.com:pparitoshh/alembic.git alembic-dev
cd ~/alembic-dev
```

---

## Work on `DAY_0_PG`

```bash
git fetch origin --prune
git switch --track origin/DAY_0_PG
```

If the branch is already local:

```bash
git switch DAY_0_PG
```

Then:

```bash
git pull --ff-only origin DAY_0_PG
```

---

## Commit and push

```bash
git status
git add .
git commit -m "your commit message"
git push
```

---

# 38. Final Day 0 checklist

Before starting development, make sure:

```text
[ ] I received my Leonardo username and password

[ ] I can log in using:
    ssh a08trcXX@login01-ext.leonardo.cineca.it

[ ] I created my laptop Leonardo SSH key

[ ] I copied my laptop public key to Leonardo

[ ] `ssh leonardo` works

[ ] My Git name is configured

[ ] My Git email is configured

[ ] I can access:
    /leonardo_work/EUHPC_D30_031/alembic/alembic

[ ] SBATCH_ACCOUNT is:
    EUHPC_D30_031

[ ] REPO is:
    /leonardo_work/EUHPC_D30_031/alembic/alembic

[ ] I know the models and heavy environments are already shared

[ ] I did NOT redownload all models

[ ] DistillKit tests pass

[ ] I created a GitHub SSH key on Leonardo

[ ] `ssh -T git@github.com` successfully authenticates me

[ ] I created my own clone in:
    ~/alembic-dev

[ ] I am on the correct Git branch

[ ] I know GPU workloads must be submitted through SLURM
```

Once these are complete:

```text
DAY 0 LEONARDO SETUP COMPLETE ✅
```

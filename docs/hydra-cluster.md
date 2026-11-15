<!-- Copy of the Hydra cluster documentation (TU Berlin ML group), saved 2026-11-14 for reference.
     Internal links (e.g. /cheating-card.md, /file-handling/apptainer) point to the original
     documentation and do not work here. Our own cluster guide: cluster/README.md -->

# Documentation of the hydra cluster

Hydra is the name of the high performance computing cluster of the ML, MLSEC, UNIML, and Cognition research groups at TU Berlin.
It consists of many dedicated heads (i.e. nodes) for GPU and CPU tasks and utilizes the open source workload manager [SLURM](https://slurm.schedmd.com/overview.html).
Here you can find all the relevant information to get started and learn about best practices.
In the [cheating card](/cheating-card.md) you can quickly look up everything about Slurm.

## Contents

+ [Specs](/README.md#Specs)
+ [How to get access](/README.md#how-to-get-access)
+ [File handling](/README.md#file-handling)
    + [/home](/README.md#home)
    + [/tmp and /temp](/README.md#tmp-and-temp)
    + [/archive](/README.md#archive)
    + [/projects](/README.md#project)
+ [Setting up your environment](/README.md#setting-up-your-environment)
+ [Submitting jobs](/README.md#submitting-jobs)
    + [Available partitions](/README.md#available-partitions)
+ [Additional infos](/README.md#additional-infos)
    + [Statistics](/README.md#seff)
    + [Cancelling jobs](/README.md#howto-cancel-slurm-jobs)
    + [Transitioning from SGE](/README.md#transitioning-from-sge)
    + [BeeOND](/README.md#beeond)
    + [Backup](/README.md#backup)
    + [Jupyter Notebook](/README.md#jupyter-notebook)
    + [Who is who](/README.md#who-is-who)
    + [Agent accounts](/README.md#agent-accounts)
+ [Getting help](/README.md#getting-help)


## Specs

| Number of heads | CPU | RAM | GPU | <a name="GPUTYPE">Slurm gpu type</a> | Fast local storage | Name |
|--|--|--|--| -- | -- | -- | 
| 20 | 2x16 | 512 GB | - | - | 1.7 TB | head001-020 |
| 1 | 2x16 | 1024 GB | 8x A100 80GB Partitioned with MIG - 16 virtual GPUs | mig40 | 7.3 TB | head022 |
| 4 | 2x16 | 1024 GB | 8x A100 80GB | 80gb | 7.3 TB | head021; head023-025 |
| 2| 2x16 | 750 GB | 2x Quadro RTX 6000 24 GB | 6000 | 3,5 TB | head034-035 |
| 1 | 4x18 | 6000 GB | - | - | 3,5 TB | head036 |
| 12 | 2x16 | 775 GB | 4x A100 40GB | 40gb | 3.2 TB | head040-051 |
| 4 | 2x10 | 180 GB | 2x RTX 3090 24GB | 3090 | 1.3 TB | head055, 058| 
| 1 | 2x32 | 775 GB | 4x NVIDIA H200 | h200 | 3.5 TB | head056, head059 |
| 1 | 2x64 | 775 GB | 2x RTX PRO 6000 Blackwell 97GB | blackwell | 14 TB | head057 |
| 7 | 2x56 | 2048 GB | - | - | 12.8 TB | head065-071 | 
| 1 | 2x28 | 1024 GB | - | - | 3.8 TB | head072 | 
| 4 | 2x16 | 768 GB | 4x H100 80GB  | h100 | 3.8 TB | head073 - head076|
| 4 | 2x16 | 768 GB | 4x A100 80GB  | 80gb | 3.8 TB | head077 - head080|



## How to get access

The cluster can only be accessed via ssh. 
If authentification with ssh is new to you, you can follow [this tutorial](https://git-scm.com/book/en/v2/Git-on-the-Server-Generating-Your-SSH-Public-Key) to create a new pair of public and private keys.
To gain access, please send an e-mail with the following information to the cluster admin:

+ Your full **name**
+ Your desired **username** on the cluster
+ Your **public ssh key** (from the file ending with .pub)

After you receive a confirmation, you can connect to the main node of the cluster as follows:

```
ssh username@hydra.ml.tu-berlin.de
```

Just as the crab Karkinos came to the aid of the mythical Hydra, there is now a second entry node, `karkinos`:

```
ssh username@karkinos.ml.tu-berlin.de
```

The main node provides a very restricted environment from which you can submit more demanding jobs, e.g. training of a neural network, to the other heads of hydra.
Please do not run any computationally heavy tasks on the main node.
Instead, follow our guides below to set up your environment and to learn how to submit jobs that take advantage of our powerful computational resources.

### Weekly security reboots

Both entry nodes automatically install security updates and reboot once a week.

| Entry node | Reboot |
|--|--|
| `hydra` | every Monday at 01:00 |
| `karkinos` | every Thursday at 01:00 |

A reboot terminates all ssh, `tmux` and `screen` sessions as well as all other processes running on that entry node, so save your work beforehand.

### Lazy access to hydra

To save keyboard strokes and not always having to type the full hostname you can modify the `~/.ssh/config` file on your local computer.
Then this command is enough to connect to Hydra: ```ssh hydra ``` (or ```ssh karkinos ``` for the second entry node).

```
Host hydra
   HostName hydra.ml.tu-berlin.de
   ForwardAgent yes
   IdentityFile ~/.ssh/id_rsa
   User <username>

Host karkinos
   HostName karkinos.ml.tu-berlin.de
   ForwardAgent yes
   IdentityFile ~/.ssh/id_rsa
   User <username>
```

## File handling

The filesystem of Hydra is a [BeeGFS](https://www.beegfs.io/), it consists of 4 metadata servers and 7 storage servers.
All these servers are connected to the main switch with 100GbE.

### /home

All data under `/home` is available on all heads.

**Each user may store at most 500 GB in their home directory.**
If you exceed this limit, your SLURM priority will be lowered.
Put your project data in a shared project directory under `/home/projects` instead (see [Project](#project)).

Since `/home` is a shared filesystem, there are some characteristics that your local filesystem does not have.

* avoid many small files
    * an ``ls`` in a directory with 2k files will also generate 2k network requests and take a long time to complete
* streaming large files is faster
    * containerize as much as possible
        * use [squashfs](/file-handling/squashfs) for datasets
        * use [Apptainer](/file-handling/apptainer) for your environment 


### /tmp and /temp

If a dataset does not fully fit into RAM, batches need to be read repeatedly from the filesystem.
Doing this on `/home` is extremely slow and will significantly impair the training time.
Therefore, each head has fast, temporary SSD storage where you should copy the [containerized datasets](/file-handling/squashfs) before training.
There are two locations for temporary data on the heads: 

* `/tmp` is virtual and only valid for one job, i.e. you can always copy stuff here and it will automatically be removed once the job finishes
* `/temp` is permanent, but it will be deleted after 7 days (can be extended with `touch /temp/XXX` on the head)

### /archive
`/archive` is a large-capacity, low-cost filesystem optimized for long-term data storage. Performance is limited due to the cost-effective storage tier.

* `Target directory:` /archive/YEAR/PROJECT (e.g., /archive/2025/foo) 


## Project

With HyMa you create shared project directories under `/home/projects` and decide yourself who may work in them.
All commands are run on the main node with `hyma -p ...`.

Prefer small, focused projects over one large one.
**Projects that have not been written to for six months are moved to [/archive](#archive), following the scheme `/archive/YEAR/PROJECT`, where `YEAR` is the year the project was last written to.**
This keeps `/home` tidy.


### Concepts

Projects live under `/home/projects/UMBRELLA_PROJECT/PROJECT`.
The **umbrella project** refers to your research group, field, or a category (e.g., `ml`, `qcml`, or `omics`), the **project** is the directory you work in.
Every project has its own Unix group `UMBRELLA_PROJECT-PROJECT`, and only its members can read and write the directory.

+ Whoever creates a group is its **owner**.
+ Every **member** can add and remove other members of the same group.
+ Being a member of an umbrella project does **not** give access to its projects, it only allows you to create new projects under it (unless a project follows its umbrella project, see [Sharing a project with the whole umbrella project](#sharing-a-project-with-the-whole-umbrella-project)).

Names must start with a lowercase letter and may only contain `a-z`, `0-9`, `-` and `_`.


### Creating a project

```
hyma -p create ml/llm-eval --description "Evaluation of large language models"
```

If the umbrella project does not exist yet, it is created and you become its owner.
If it already exists, only its members can create projects under it. Ask one of them (`hyma -p members ml`) to add you.
Please check with `hyma -p list` whether your research group already has an umbrella project before you start a new one.
`hyma -p list --me` shows only the projects you are a member of.

Each user can own at most 25 projects and 3 umbrella projects.


### Managing members

```
hyma -p add-member ml/llm-eval dave      # access to the project
hyma -p add-member ml dave               # may create projects under ml
hyma -p remove-member ml/llm-eval dave
```

The owner cannot be removed. To leave, hand the group over first:

```
hyma -p change-owner ml/llm-eval bob
```

> New memberships take effect at your next login. Log out and in again and check with `id`.


### Sharing a project with the whole umbrella project

If everyone in the umbrella project should work in a project, you don't have to add them one by one:

```
hyma -p create ml/llm-eval --inherit-members   # copy the current members of ml once
hyma -p create ml/llm-eval --sync-members      # members always stay the same as in ml
```

With `--inherit-members` the members of `ml` are copied once, later changes to `ml` are not carried over.
With `--sync-members` the project **follows** its umbrella project: whoever is added to or removed from `ml` is added to or removed from the project as well.
The members of such a project are then only managed on the umbrella project, so use `hyma -p add-member ml dave` instead of `hyma -p add-member ml/llm-eval dave`.

The owner can switch this on or off later:

```
hyma -p sync ml/llm-eval --enable
hyma -p sync ml/llm-eval --disable    # the current members stay
```

`--enable` only works if every member of the project is also a member of the umbrella project, otherwise hyma tells you who has to be removed first.
As long as you own a project that follows its umbrella project, you cannot leave the umbrella project, and you can only hand the project over to a member of the umbrella project.

If `hyma -p show` reports the status `dirty`, a change to the umbrella project was interrupted before it reached the project.
Any member of the umbrella project can finish it with:

```
hyma -p sync ml
```


### Fixing permissions

Files that you copy in with `cp -p`/`rsync -a`, move in with `mv` or unpack with `tar -p` keep their old group, so your colleagues cannot read them.
Any member can repair the whole project with:

```
hyma -p fix-perms ml/llm-eval
```


### Command overview

| Command | Description | Who |
| -- | -- | -- |
| `create UMBRELLA/PROJECT [--description TEXT] [--inherit-members] [--sync-members]` | create a project | anyone for a new umbrella project, otherwise its members |
| `list [UMBRELLA] [--me]` | list projects (`--me`: only those you are a member of) | everyone |
| `show UMBRELLA[/PROJECT]` | show owner, members, description | everyone |
| `members UMBRELLA[/PROJECT]` | list members | everyone |
| `add-member UMBRELLA[/PROJECT] USER` | add a member | members |
| `remove-member UMBRELLA[/PROJECT] USER` | remove a member | members |
| `change-owner UMBRELLA[/PROJECT] USER` | hand the group over | owner |
| `edit UMBRELLA[/PROJECT] --description TEXT` | change the description | members |
| `fix-perms UMBRELLA/PROJECT` | repair group and permissions | members |
| `sync UMBRELLA/PROJECT --enable\|--disable` | let a project follow its umbrella project, or stop | owner |
| `sync UMBRELLA[/PROJECT]` | carry over an interrupted change (status `dirty`) | members of the umbrella project |

`hyma -p --help` shows the built-in help.


## Setting up your environment

Scientists in our groups are working on different applications and therefore require different software to run their experiments.
This usually means that you have to set up your own environments where you install specific packages.

To efficiently set up environments, we use `apptainer`.
Just follow our [simple tutorial with example configurations](/file-handling/apptainer). 
It allows you to package software applications and all of their dependencies into a single image that can be run uniformly across different hardware.
This keeps our file-io lightweight by avoiding many small files, which improves the cluster experience for all users. 
You can easily share your containers with others, move them between systems, or even upload them to a public repository.
We also have a few [pre-built containers](/file-handling/apptainer/README.md#pre-built-containers) available on the cluster that you can use for testing purposes or if you only require very basic packages.

Similarly, we store datasets that consist of many individual files as SquashFS images.
You can simply follow our [tutorial on SquashFS](/file-handling/squashfs) in order to learn how to containerize your data and use it within your `apptainer` environments.


## Submitting jobs

The purpose of the cluster is to run our experiments and simulations in the most effective and fairest way.
To this end, you connect to the cluster and submit your jobs from the main head using the command `sbatch`.
The SLURM scheduler will then run the jobs on the compute heads as soon as the requested resources are available.

Jobs are defined in a bash `.sh` script.
It contains the commands that you want to execute, e.g. a python script that trains a neural network, and additional `#SBATCH` directives for the SLURM scheduler.
These options appear at the top of your script and usually define the hardware requirements and other settings for the job. 
Here is a simple example for a gpu job that runs a python script in your [containerized environment](/file-handling/apptainer):

```sh
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv python_container.sif python example_script.py
```
It contains the most important options, which are:

- `job_name`: specifies the name of the job
- `partition`: the partition of the job (see [available partitions below](/README.md#available-partitions))
- `gpus-per-node`: number of GPUs per node. 
- `constraint`: specifies the gpu-type by [[Slurm gpu type](#GPUTYPE)]. Example for one A100 with 80GB or 40GB GPU: `--constraint="80gb|40gb"`
- `ntasks-per-node`: number of CPUs per node
- `output`: path for the log files, where `%j` is replaced with the job id by SLURM

Further information and example scripts can be found [here](/sbatch-examples/).
When you submit your job, you can override options from the script or provide new ones directly in the command line.
For example, you can provide your e-mail address to be notified when the job starts, ends or fails:

```shell
sbatch --mail-type=BEGIN,FAIL,END --mail-user=<you>@tu-berlin.de <path-to-slurm-job>.sh
```
The number of possible options is tremendous. Please check the [official SLURM documentation](https://slurm.schedmd.com/sbatch.html#SECTION_OPTIONS) for more details.

To monitor whether your jobs are running or waiting in the queue, use:

```
squeue -u <username>
```

The normal method to kill a SLURM job is:

```
scancel <job-id>
```


### Available partitions

You have to select an appropriate partition for your job in the options of your script.
We have several partitions for cpu and gpu jobs that have different runtimes as listed in the table below.
The runtimes range from 9 minutes to 7 days and jobs are killed after exceeding it.
Users have less slots on the partitions with long runtimes.
Furthermore, the priority of the partition is related to the runtime of the job, i.e. the shorter the job the more priority it gets from the scheduler.
Therefore, you should try to select the partition with the shortest runtime suitable for your job.
For the CPU partitions over 2h runtime, we use [Gang Scheduling](https://slurm.schedmd.com/gang_scheduling.html) to get a better cluster utilization.
This means that the partitions are overloaded and the jobs are divided into 30 second slices that are suspended alternately.

Have you ever submitted a job to the 7 day partition and waited several hours for it to start, only to see that the job fails after 2 minutes due to an import error or a typo in your script?
To prevent this kind of frustration, there are two special partitions, `gpu-test` and `cpu-test`, where you can run tests of your scripts or environments before you send them to the other partitions.
Special rules apply to the test partitions: One running job per user and only 15 minutes runtime.
This ensures that there are always enough slots for functional tests.


| Name | Kind | Runtime | Running jobs per account | 
| -- | -- | -- | -- | 
| cpu-9m | CPU | 9m | 200 | 
| cpu-2h | CPU | 2h | 100 | 
| cpu-5h | CPU | 5h | 80 | 
| cpu-2d | CPU | 2d | 50 |
| cpu-7d | CPU | 7d | 30 |
| gpu-9m | GPU | 9m | 100 | 
| gpu-2h | GPU | 2h | 50 | 
| gpu-5h | GPU | 5h | 35 | 
| gpu-2d | GPU | 2d | 28 | 
| gpu-7d | GPU | 7d | 3 | 
| cpu-test | CPU | 15m | 1 | 
| gpu-test | GPU | 15m| 1 | 


## Additional infos

### Seff

The command seff provides statistics related to the efficiency of resource usage by the completed job.

```
seff <jobid>
```


### Howto cancel SLURM jobs

The normal method to kill a Slurm job is:
```
scancel <job-id>
```
You can find your job id with the following command:
```
squeue -u $USER
```
If the the job id is 1234567 then to kill the job:
```
scancel 1234567
```

### Transitioning from SGE

If you are transitioning from our old, SGE-based cluster, you can find an extensive comparison between SGE and SLURM [here](https://github.com/aws/aws-parallelcluster/wiki/Transition-from-SGE-to-SLURM). It lists many commands and environment variables that you might know from SGE and how these are called in SLURM.

### BeeOND

If you want to access a dataset in a job that runs on multiple heads you can generate a [common network filesystem](http://www.beegfs.io/wiki/BeeOND). This filesystem is located on the SSDs of the heads, making it much faster than `/home`, and it exists only for the duration of the job. You can find more details in our [documentation](/file-handling/beeond).

### Backup

Almost the complete filesystem is backed up once a week, all files are stored in 3 revisions on the backup server.

If you accidentally deleted something, please contact the cluster admin.

The following exceptions still exist (a new backup storage has been ordered to include everything in the future):
* `/home/space`

### Jupyter Notebook

1. Modify environment requirements: Add notebook to the requirement.txt file.
2. Create and build an apptainer container with the modified requirement.txt file as described in the [tutorial](/file-handling/apptainer/README.md#building-a-container).
3. Create job script (e.g. run_jupyter_notebook.sh):
```sh
#!/bin/bash
#SBATCH --job-name=cpu_job
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/test-jupyter-with-data-%j.out

# 1. copy the squashed dataset to the nodes /tmp 
cp /home/space/datasets-sqfs/squashfs_example.sqfs /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/squashfs_example.sqfs:/input-data:image-src=/ ./python_container.sif jupyter notebook --ip 0.0.0.0 --no-browser
```

4. Run script: ``` sbatch run_jupyter_notebook.sh ```
Note the head node (headxyz) on which jupyter is running
Open another terminal and run:
```
ssh -L 8888:headxyz:8888 -o ServerAliveInterval=60 [username]@hydra.ml.tu-berlin.de
```
Go to the logs output of the running job (job running jupyter notebook) and copy lower link and paste it to browser of your choice.
```
    [C 2023-08-29 15:37:24.524 ServerApp] 
        [...]
        Or copy and paste one of these URLs:
            <http://head002:8888/tree?token=76fa9095ec829f963fa0b51a2db489b7ad58a6915627fa25>
            <http://127.0.0.1:8888/tree?token=76fa9095ec829f963fa0b51a2db489b7ad58a6915627fa25>
```

### Who is who

With the command “cwho” on hydra you can find out who is hiding behind which username.
```
hydra:~:% cwho dominik
Name: Dominik Kuehne
Email: dominik@hydra
```

### TODO

+ use sftp to access files


## Example Setup

An example experiment setup for the Slurm cluster including a clean configuration management using [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) (unrelated to our cluster) and functionality to abstract away most of the Slurm and Apptainer commands can be found [here](https://github.com/marvinsxtr/ml-project-template) or [here](https://github.com/mx-e/ml-project-template).

You can simply copy this template to get started.

## Agent accounts

With HyMa you can create agent accounts: separate logins for AI coding agents and other automated tools.
An agent account is locked in. By default it only sees its own home and `/tmp`, and you decide explicitly which projects and directories it may use.
All commands are run on the main node with `hyma -u ...`.

Use an agent account instead of your own for anything that runs on its own.
If the agent does something wrong, it can only touch what you have opened up for it.


### Concepts

+ An agent account belongs to you, its **owner**, and is called `USERNAME-agentNN`, e.g. `bob-agent00`.
+ Only you can configure, stop or remove your agents.
+ An agent never gets more than you have: it only gets into projects you are a member of, and only into the directories you open up for it.
+ If your account is disabled, your agents are disabled as well.

Each user can have at most 2 agent accounts.

What an agent can do by default:

| | Agent |
| -- | -- |
| its own home `/home/agents/USERNAME-agentNN` | read and write |
| `/tmp`, `/var/tmp` | only its own files |
| programs and system files (`/usr`, `/etc`, ...) | read and run |
| `/home` (your home, other homes, projects), `/archive` | no access |
| Slurm | no access |
| RAM | at most 5 GB |

An agent cannot use `sudo`, `cron`, `at`, containers (Apptainer) or ssh port forwarding, and it cannot run `hyma` itself.


### Creating an agent account

```
hyma -u agent-create --description "Claude Code for llm-eval"
```

This creates your next agent account, e.g. `bob-agent00`.
Agent accounts have no password, they log in with an ssh key.
Create a separate key pair for the agent (not your own key) and add the public key:

```
ssh-keygen -t ed25519 -f ~/.ssh/agent_ed25519
hyma -u agent-add-key bob-agent00 --key-file ~/.ssh/agent_ed25519.pub
ssh -i ~/.ssh/agent_ed25519 bob-agent00@hydra.ml.tu-berlin.de
```

> A new agent can log in about a minute after it was created.

As for your own account, an entry in `~/.ssh/config` on the computer the agent runs on saves typing; then `ssh hydra-agent` is enough:

```
Host hydra-agent
   HostName hydra.ml.tu-berlin.de
   IdentityFile ~/.ssh/agent_ed25519
   User bob-agent00
```


### Giving access to projects

```
hyma -u agent-grant bob-agent00 ml/llm-eval
hyma -u agent-revoke bob-agent00 ml/llm-eval
```

You have to be a member of the project yourself, and it has to be a project, not an umbrella project.
The agent can then read and write the project like any other member.

+ If you leave a project, your agents leave it as well.
+ Every member of a project can remove an agent from it: `hyma -p remove-member ml/llm-eval bob-agent00`.


### Giving access to other directories

```
hyma -u agent-allow bob-agent00 /home/bob/code              # read only
hyma -u agent-allow bob-agent00 /home/bob/scratch --write   # read and write
hyma -u agent-disallow bob-agent00 /home/bob/code
```

Any directory can be opened up, except `/` itself.
Paths may only contain letters, digits and `.`, `_`, `+`, `-`.
The normal file permissions still apply on top: the agent can only use what its permissions allow, so it never reads anything you couldn't read yourself.
Symbolic links only work if their target is opened up as well.

**Directories in your home.** The agent is added to your own group, and your home gets `g+x` (`g+rx` if you open up your whole home).
Inside, the agent can use what your files allow your group:

```
chmod -R g+rX ~/code       # the agent may read
chmod -R g+w ~/scratch     # the agent may write (together with --write)
```

Everything group-readable in an opened directory is readable for the agent.
Files that only you can read, like `~/.ssh`, stay protected.

> Opened and closed directories take effect within a minute, also for running processes.
> Project access and access to your home take effect at the agent's next login.


### Slurm

By default an agent cannot use Slurm at all. You can allow read-only access:

```
hyma -u agent-slurm bob-agent00 read    # squeue, sinfo, sacct, scontrol show
hyma -u agent-slurm bob-agent00 off
```

With `read`, the agent can look at the queue, the nodes and finished jobs, but it cannot submit jobs.


### Stopping and removing an agent

```
hyma -u agent-kill bob-agent00            # stop all its processes on the main node
hyma -u agent-kill bob-agent00 --force    # if that doesn't help
hyma -u agent-remove bob-agent00
```

**Removing an agent deletes its home directory with everything in it.**
Copy what you want to keep into a project first.


### Command overview

| Command | Description | Who |
| -- | -- | -- |
| `agent-create [--description TEXT]` | create your next agent account | everyone |
| `agent-list` | your agents, their projects, directories and Slurm access | owner |
| `agent-show AGENT` | details of an agent, including its ssh keys | owner |
| `agent-add-key AGENT --key-file FILE` | add an ssh public key | owner |
| `agent-grant AGENT UMBRELLA/PROJECT` | give access to a project | owner, member of the project |
| `agent-revoke AGENT UMBRELLA/PROJECT` | take access to a project away | owner |
| `agent-allow AGENT PATH [--write]` | open up a directory, read only unless `--write` | owner |
| `agent-disallow AGENT PATH` | close a directory again | owner |
| `agent-slurm AGENT read\|off` | read-only Slurm access, or none | owner |
| `agent-kill AGENT [--force]` | stop all processes of the agent | owner |
| `agent-remove AGENT` | remove the agent and its home directory | owner |

`hyma -u --help` shows the built-in help.

## Getting help

If you couldn't find what you needed here (e.g. you want to do something not covered in the documentation or you require additional packages/resources), please contact the cluster team before hacking your own solutions.
We are happy to assist you and extend this documentation in order to keep the cluster experience as smooth as possible for everyone.

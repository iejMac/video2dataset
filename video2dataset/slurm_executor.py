import os
import time
import yaml
import subprocess
from datetime import datetime

import fsspec
import fire


class SlurmDistributor:
    """Parallelism via slurm"""
    def __init__(self,
                 worker_args,
                 cpus_per_task,
                 job_name,
                 partition,
                 n_nodes,
                 gpus_per_node,
                 account,
                 tasks_per_node=1,
                 nodelist=None,
                 exclude=None,
                 cache_path=None,
                 timeout=None,
                 verbose_wait=False
                 ):
        from .main import make_tmp_dir, make_path_absolute
        self.cpus_per_task = cpus_per_task
        self.job_name = job_name
        self.partition = partition
        self.n_nodes = n_nodes
        self.gpus_per_node = gpus_per_node
        self.account = account
        self.tasks_per_node = tasks_per_node
        self.nodelist = nodelist
        self.exclude = exclude
        self.cache_path =cache_path
        if not cache_path:
            cache_path = '.video2dataset_cache/'
        self.timeout = timeout
        self.verbose_wait = verbose_wait

        self.fs, self.cache_path = fsspec.core.url_to_fs(cache_path)
        if not self.fs.exists(self.cache_path):
            self.fs.mkdir(self.cache_path)

        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # change distributor type for the subprocesses
        worker_args['distributor'] = 'multiprocessing'

        output_folder = worker_args.get('output_folder', 'video')
        output_folder = make_path_absolute(output_folder)
        self.fs, self.tmp_path, _ = make_tmp_dir(output_folder)



        # save worker args to file (this is written by the slurm_executor)
        self.worker_args_as_file = os.path.join(self.cache_path, f'{self.timestamp}_worker_args.yaml')
        with self.fs.open(self.worker_args_as_file, 'w', encoding="utf-8") as f:
            yaml.dump(worker_args, f, default_flow_style=False)

        self.launcher_path = os.path.join(self.cache_path, self.timestamp + f'_launcher.sh')
        with self.fs.open(self.launcher_path, 'w', encoding="utf-8") as launcher_file:
            launcher_file.write(self._make_launch_cpu())

        self.sbatch_path = os.path.join(self.cache_path, self.timestamp + f'_sbatch.sh')
        with self.fs.open(self.sbatch_path, 'w', encoding="utf-8") as sbatch_file:
            sbatch_file.write(self._make_sbatch())

        print(f'Wrote launcher to {self.launcher_path}')
        print(f'Wrote sbatch to {self.sbatch_path}')

    def _make_sbatch(self
            ):

        nodelist = ("#SBATCH --nodelist " + self.nodelist) if self.nodelist is not None else ""
        exclude =  ("#SBATCH --exclude " + self.exclude) if self.exclude is not None else ""
        account = ("#SBATCH --account " + self.account) if self.account is not None else ""
        return f"""#!/bin/bash
#SBATCH --partition={self.partition}
#SBATCH --job-name={self.job_name}
#SBATCH --output={self.cache_path}/slurm-%x_%j.out
#SBATCH --nodes={self.n_nodes}
#SBATCH --ntasks-per-node={self.tasks_per_node}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --gpus-per-node={self.gpus_per_node}
#SBATCH --exclusive
{nodelist}
{exclude}
{account}
#SBATCH --open-mode append

srun --account {self.account} bash {self.launcher_path}

"""



    def _make_launch_cpu(self,):

        venv = os.environ['VIRTUAL_ENV']
        path2self = os.path.abspath(__file__)
        cdir = '/'.join(path2self.split('/')[:-1])
        project_root = os.path.abspath(os.path.join(cdir, '..'))
        return f"""#!/bin/bash
# mpi version for node rank
H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
export NODE_RANK=${{THEID}}
echo THEID=$THEID

# set new location for cache dir
export XDG_CACHE_HOME="{self.cache_path}"


cd {project_root}
source {venv}/bin/activate

python {path2self} --worker_args {self.worker_args_as_file} --node_id $SLURM_NODEID --n_nodes $SLURM_JOB_NUM_NODES --num_tasks_per_node $SLURM_NTASKS_PER_NODE --subtask_id $SLURM_LOCALID
"""

    def __call__(self,*args,**kwargs):
        print(f'{self.__class__.__name__} starting the job')

        status = self._run_job()

        # interpret the results
        if status == "success":
            print("job succeeded")
            return True
        elif status == "failed":
            print("job failed")
            return False
        else:
            print("exception occurred")
            return False

    def _start_job(self, sbatch_file):
        """start job"""
        args = ["sbatch"]
        args.append(sbatch_file)
        sbatch_output = subprocess.check_output(args).decode("utf8")
        lines = sbatch_output.split("\n")

        lines = [line for line in lines if "Submitted" in line]
        if len(lines) == 0:
            raise ValueError(f"slurm sbatch failed: {sbatch_output}")

        parsed_sbatch = lines[0].split(" ")
        job_id = parsed_sbatch[3].strip()
        return job_id

    def _run_job(self,):
        """
        Run a job and wait for it to finish.
        """
        try:
            job_id = self._start_job(self.sbatch_path)

            print(f"waiting for job {job_id}")

            timeout = self.timeout

            if timeout is None:
                print("You have not specified a timeout, defaulting to 2 weeks.")
                timeout = 1.21e6

            status = self._wait_for_job_to_finish(job_id=job_id, timeout=timeout)

            if not status:
                print(f"canceling {job_id}")
                subprocess.check_output(["scancel", job_id]).decode("utf8")
                status = self._wait_for_job_to_finish(job_id)
                print("job cancelled")
                return "failed"
            else:
                print("job succeeded")
                return "success"
        except Exception as e:  # pylint: disable=broad-except
            print(e)
            return "exception occurred"

    def _wait_for_job_to_finish(self, job_id, timeout=30):
        t = time.time()
        while 1:
            if time.time() - t > timeout:
                return False
            time.sleep(1)
            if self._is_job_finished(job_id):
                return True

    def _is_job_finished(self, job_id):
        status = subprocess.check_output(["squeue", "-j", job_id]).decode("utf8")

        if self.verbose_wait:
            print(f"job status is {status}")

        return status == "slurm_load_jobs error: Invalid job id specified" or len(status.split("\n")) == 2


class ShardSampler:
    """
        Should be callable to select samples based on the node_id
        :param global_task_id: The global task id for the current task
        :param num_tasks: The overall number of tasks
        :return:
    """
    def __init__(self, global_task_id,num_tasks):
        self.task_id = global_task_id
        self.num_tasks = num_tasks

    def __call__(self,shardfile_list):
        shardlist = [(full_shard_id, shard_id) for full_shard_id, shard_id in shardfile_list if shard_id % self.num_tasks == self.task_id]
        return shardlist


def executor(worker_args, node_id, n_nodes, num_tasks_per_node, subtask_id):
    from video2dataset import video2dataset

    num_tasks = n_nodes * num_tasks_per_node
    print('#' * 100)
    print("args:")
    print(worker_args)
    print('node id', node_id)
    print('n_nodes', n_nodes)
    print('num_tasks_per_node', num_tasks_per_node)
    print('subtask_id', subtask_id)
    print('num_tasks', num_tasks)
    print('#' * 100)

    global_task_id = node_id * num_tasks_per_node + subtask_id
    assert global_task_id < num_tasks, f'global_task_id is {global_task_id} but must be less than num_nodes*num_tasks_per_node={n_nodes * num_tasks_per_node}'

    print(f'Starting task with id {global_task_id}')
    os.environ['GLOBAL_RANK'] = str(global_task_id)


    # Read the worker args from the file
    with open(worker_args, "r", encoding="utf-8") as worker_args_file:
        worker_args = yaml.load(worker_args_file,Loader=yaml.SafeLoader)
    sampler = ShardSampler(global_task_id=global_task_id,
                           num_tasks=num_tasks)

    worker_args.pop('sampler', None)
    # call main script from every subprocess
    video2dataset(sampler=sampler,
                  **worker_args)

if __name__ == '__main__':
    fire.Fire(executor)
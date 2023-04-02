"""Executor for distribution via slurm"""

import os
import yaml
import fire

from video2dataset import video2dataset


class ShardSampler:
    """
    Should be callable to select samples based on the node_id
    :param global_task_id: The global task id for the current task
    :param num_tasks: The overall number of tasks
    :return:
    """

    def __init__(self, global_task_id, num_tasks):
        self.task_id = global_task_id
        self.num_tasks = num_tasks

    def __call__(self, shardfile_list):
        shardlist = [
            (full_shard_id, shard_id)
            for full_shard_id, shard_id in shardfile_list
            if int(full_shard_id) % self.num_tasks == self.task_id
        ]
        return shardlist


def executor(worker_args, node_id, n_nodes, num_tasks_per_node, subtask_id):
    """
    Spins up the individual workers on the nodes
    :param worker_args: parameters to video2dataset on the respective node
    :param node_id: node id
    :param n_nodes: overall number of node
    :param num_tasks_per_node: number of distinct calls to video2dataset from the current node
    :param subtask_id: local id of current call to video2dataset
    :return:
    """

    num_tasks = n_nodes * num_tasks_per_node
    print("#" * 100)
    print("args:")
    print(worker_args)
    print("node id", node_id)
    print("n_nodes", n_nodes)
    print("num_tasks_per_node", num_tasks_per_node)
    print("subtask_id", subtask_id)
    print("num_tasks", num_tasks)
    print("#" * 100)

    global_task_id = node_id * num_tasks_per_node + subtask_id
    assert global_task_id < num_tasks, (
        f"global_task_id is {global_task_id} but must be less than "
        f"num_nodes*num_tasks_per_node={n_nodes * num_tasks_per_node}"
    )

    print(f"Starting task with id {global_task_id}")
    os.environ["GLOBAL_RANK"] = str(global_task_id)
    os.environ["LOCAL_RANK"] = str(subtask_id)

    # Read the worker args from the file
    with open(worker_args, "r", encoding="utf-8") as worker_args_file:
        worker_args = yaml.load(worker_args_file, Loader=yaml.SafeLoader)
    sampler = ShardSampler(global_task_id=global_task_id, num_tasks=num_tasks)

    worker_args.pop("sampler", None)
    # call main script from every subprocess
    video2dataset(sampler=sampler, **worker_args)


if __name__ == "__main__":
    fire.Fire(executor)

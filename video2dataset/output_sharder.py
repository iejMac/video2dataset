"""Reader is module to read the url list and return shards"""
import braceexpand

from multiprocessing.pool import ThreadPool


class OutputSharder:
    """
    The reader class reads a shard list and returns shards

    It provides an iter method
    It provides attributes:
    - shard_list: a list of shards to read
    - input_format: the format of the input dataset
    - done_shards: a set of already done shards
    - group_shards: the number of shards to group together 
    """
    def __init__(
        self,
        shard_list,
        input_format,
        done_shards,
        tmp_path,
        group_shards=1,
    ) -> None:

        self.input_format = input_format
        self.done_shards = done_shards
        self.group_shards = group_shards # TODO: use this to make the shape of shard list [-1, group_shards]
        self.shard_list = list(braceexpand.braceexpand(shard_list))

        if self.input_format == "webdataset":
            self.shard_ids = [s.split("/")[-1][:-len(".tar")] for s in self.shard_list]
        elif self.input_format == "files":
            self.shard_ids = [s.split("/")[-1] for s in self.shard_list]

        # TODO: this should be array of arrays of (s_id, s) pairs the second of which is group_shards size
        # for now just hacking to sizze 1
        self.shards = [[s] for s_id, s in zip(self.shard_ids, self.shard_list) if s_id not in self.done_shards]

    def __iter__(self):
        """
        Iterate over shards, yield shards of size group_shards size
        Each shard is a tuple (shard_id, shard)
        """
        for shard_ids, shards in self.shards:
            yield (shard_ids, shards)

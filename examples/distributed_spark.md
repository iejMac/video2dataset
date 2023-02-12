# video2dataset distributing with pyspark

Using PySpark for distributing work in video2dataset is preferred for larger datasets. Here's how we used it on a slurm cluster to download and process 40M youtube videos.

## Setup

Download and extract spark via ```wget https://dlcdn.apache.org/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz && tar xf spark-3.3.1-bin-hadoop3.tgz```.

## Creating the spark server

To create the spark server you need to run the following sbatch script via ```sbatch```

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=spark_on_slurm
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0 # 0 means use all available memory (in MB)
#SBATCH --output=%x_%j.out
#SBATCH --comment laion
#SBATCH --exclusive

srun --comment laion bash worker_spark_on_slurm.sh
```

Which runs this bash script:

```bash
#!/bin/bash
#
# get environment variables
GLOBAL_RANK=$SLURM_PROCID
CPUS=`grep -c ^processor /proc/cpuinfo`
MEM=$((`grep MemTotal /proc/meminfo | awk '{print $2}'`/1000)) # seems to be in MB
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
LOCAL_IP=$(hostname -I | awk '{print $1}')

# setup the master node
if [ $GLOBAL_RANK == 0 ]
then
    # print out some info
    echo -e "MASTER ADDR: $MASTER_ADDR\tGLOBAL RANK: $GLOBAL_RANK\tCPUS PER TASK: $CPUS\tMEM PER NODE: $MEM"

    # then start the spark master node in the background
    ./spark-3.3.1-bin-hadoop3/sbin/start-master.sh -p 7077 -h $LOCAL_IP
   
fi

sleep 10

# then start the spark worker node in the background
MEM_IN_GB=$(($MEM / 1000))
# concat a "G" to the end of the memory string
MEM_IN_GB="$MEM_IN_GB"G
echo "MEM IN GB: $MEM_IN_GB"

./spark-3.3.1-bin-hadoop3/sbin/start-worker.sh -c $CPUS -m $MEM_IN_GB "spark://$MASTER_ADDR:7077"
echo "Hello from worker $GLOBAL_RANK"

sleep 10

if [ $GLOBAL_RANK == 0 ]
then
    # then start some script
    echo "hi"
fi

sleep 1000000
```

## Running the video2dataset job

Once you have the slurm cluster running you can ssh into the master node and simply run the following python script adjusted for your particular use case. You might need to adjust num_cores, mem_gb, master_node IP, etc.

```python3
from video2dataset import video2dataset
import sys
import shutil
import os
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

output_dir = os.path.abspath("bench")

def aws_ec2_s3_spark_session(master, num_cores=128, mem_gb=256):
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    main_memory = str(int(mem_gb * 0.9)) + "g"
    memory_overhead = str(mem_gb - int(mem_gb * 0.9)) + "g"
    spark = (
        SparkSession.builder.config("spark.submit.deployMode", "client")
        .config("spark.executor.memory", main_memory)
        .config("spark.executor.cores", str(num_cores))  # this can be set to the number of cores of the machine
        .config("spark.task.cpus", "1")
        .config("spark.executor.memoryOverhead", memory_overhead)
        .config("spark.task.maxFailures", "2")
        .master(master)  # this should be set to the spark master url
        .appName("cc2dataset")
        .getOrCreate()
    )
    return spark


if __name__ == "__main__":
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    master_node = "IP OF THE MASTER NODE"
    spark = aws_ec2_s3_spark_session(f"spark://{master_node}:7077", num_cores=48, mem_gb=256)

    video2dataset(
        url_list="/admin/home-iejmac/datasets/acav100m/ACAV1M_clip_unique.parquet",
        output_folder="s3://s-laion/acav100m/test_acav100m",
        output_format="webdataset",
        input_format="parquet",
        url_col="videoLoc",
        caption_col="title",
        clip_col="clip",
        save_additional_columns=["description", "videoID", "start", "end"],
        enable_wandb=True,
        video_size=360,
        strict_resize=False,
        number_sample_per_shard=100,
        subjob_size=10000,
        processes_count=96,
        thread_count=48,
        distributor="pyspark",
    )
```

Once you run this the video2dataset job should be distributed among all spark workers.

## Checking on the job

You can check the output of the workers in the spark folder you untarred earier in the work or logs directories. You can also check the spark UI by doing ```ssh -L 4040:localhost:4040 -L 8080:localhost:8080 login_node``` followed by ```ssh -L localhost:4040:master_node:4040 -L localhost:8080:master_node:8080 master_node``` and checking http://localhost:4040 and http://localhost:8080 in your browser.

"""aws utls"""
import random
import subprocess

def ls_aws(path, ending="tar", max_dirs=500, **kwargs):
    if not path.endswith("/"):
        path = path + "/"

    all_files = []
    all_dirs = list(range(max_dirs))
    random.shuffle(all_dirs)

    for curr_dir in all_dirs:
        curr_path = path + f"{curr_dir:03}/"
        cmd = f"/usr/local/bin/aws s3 ls {curr_path}"
        if ending is not None:
            cmd += f" | grep {ending}"

        stdout = (
            subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
            .stdout.decode()
            .split("\n")
        )
        for line in stdout:
            if line:
                all_files.append(curr_path + line.split(" ")[-1])
    
    return all_files

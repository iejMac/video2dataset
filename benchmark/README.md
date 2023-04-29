# video2dataset Benchmark Suite
The code in here can be used to benchmark the performance of video2dataset components (and end2end) on different hardware/data conditions

## Subsamplers

Test out the performance of video2dataset subsamplers by configuring the ```subsamplers_config.yaml``` file with the subsamplers you want to benchmark along with the parameters (to form a parameter grid to test over). You can also choose a set of videos to test on (in case thats relevant for your use case). The script will output a...
# TODO: come up with what will it will output, how do we visualize it etc.


* Benchmarks for cut detector (on a 96 core machine)
    * 3.947849946894834 VIDS/S
    * 2054.9439520235096 FRAMES/S

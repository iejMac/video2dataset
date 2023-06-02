# API

The module exposes a single function `video2dataset` which takes the same arguments as the command line tool:

## Core Args

```
url_list: list of input urls - can be any of the supported input formats
    (csv, parquet, braceexpand tar paths etc.)
output_folder: Desired location of output dataset (default = "dataset")
output_format: Format of output dataset, can be (default = "files")
    - files, samples saved in subdirectory for each shard (useful for debugging)
    - webdataset, samples saved in tars (useful for efficient loading)
    - parquet, sampels saved in parquet (as bytes)
    - tfrecord, samples saved in tfrecord (as bytes)
    - dummy, does not save (useful for benchmarks)
input_format: Format of the input, can be (default = "csv")
    - txt, text file with a url in each line
    - csv, csv file with urls, (and captions + metadata)
    - tsv, tsv - || -
    - tsv.gz, - || - but compressed gzip
    - json, loads urls and metadata as json
    - parquet, loads urls and metadata as parquet
    - webdataset, usually braceexpand format of mutliple paths to tars to re-process
encode_formats: Dict that specifies what extension each modality should use (default = "{'video': 'mp4'}")
    f.e. {"video": "mp4", "audio": "m4a"}
stage: String that tells video2dataset what stage of processing is being performed. Can be (default = 'download')
    WARNING: To be depracated soon (this information should be deduced based on config)
    - download, when input is some tabular format and data must be downloaded first
    - subset, tar files are already written and we would like to re-process (input_format == "webdataset")
    - optical_flow, tar files are written and we woudl like to compute optical_flow and save to md shards
url_col: Column in input (if has columns) that contains the url (default = "url")
caption_col: Column in input (if has columns) that contains captions (to be written as txt) (default = None)
clip_col: Column in input (if has columns) that contains timeframes of clips for how to split video (default = None)
save_additional_columns: List of column names to save to json component of a sample (defualt = None)
enable_wandb: Whether or not to log info to wandb (default = False)
wandb_project: Name of wandb project to log runs to (default = "video2dataset")
incremental_mode: Decides how to handle restarting, Can be (default = "incremental")
    - incremental, checks which shards are done and skips those
    - overwrite, deletes and reprocesses shards as it goes
max_shard_retry: Maximum amount of attempts to retry a failed shard (default = 1)
tmp_dir: Path to temporary directory on your file system (default = "/tmp")
config: Path to your config of choice or the config itself (more info on configs in API doc) (default = "default")
```

## Config

Any video2dataset configs is divided into 4 components which describe the following:
- subsampling: how the data should be transformed on its way from input to output
- reading: how the data should be read (from the internet or local FS)
- storage: how the data should be stored
- distribution: how processing should be distributed among compute resources

Lets look at an example and go over where we can find information on how to fill out a config:

```yaml
subsampling:
    ResolutionSubsampler:
        args:
            video_size: 224
            resize_mode: "scale,crop,pad"
    FrameSubsampler:
        args:
           frame_rate: 5
    CutDetectionSubsampler:
        cuts_are_clips: True
        args:
            cut_detection_mode: "all"
            framerates: null
            threshold: 27
            min_scene_len: 15
    ClippingSubsampler:
        args:
            min_length: 4.0
            max_length: 20.0
            max_length_strategy: "all"
            precision: "keyframe_adjusted"
    FFProbeSubsampler:
        args:
            extract_keyframes: False

reading:
    yt_args:
        download_size: 360
        download_audio_rate: 44100
        yt_metadata_args: null
    timeout: 60
    sampler: null

storage:
    number_sample_per_shard: 1000
    oom_shard_count: 5
    captions_are_subtitles: False

distribution:
    processes_count: 16
    thread_count: 32
    subjob_size: 1000
    distributor: "multiprocessing"
```

### Subsampling

This component of the config will contain a dict with subsampler names and their respective parameters. Each subsampler specification is formatted as such:

```yaml
SubsamplerName:
    v2ds_arg1: "test"
    ...
    v2ds_argn: "test"
    args:
        ss_init_arg1: 1
        ...
        ss_init_argn: n
```

The arguments within each subsampler are split into 2 groups:
- initialization args: these are the arguments inside the "args" dict of each subsampler specification. If this is not present the subsampler will not be initialized. To check what values can be present in an "args" dict of a given subsampler check the documentation in the [docstring of a subsampler](https://github.com/iejMac/video2dataset/tree/main/video2dataset/subsamplers).
- non-initialization args: the video2dataset worker will use these to perform operations outside the subsampler. They can be used to specify processes even without the presenece of the initialized subsampler. A good example is the `cuts_are_clips` parameter in the `CutDetectionSubsampler`. If we perform a stage where we detect cuts in the input videos and place those cuts in the metadata but do not clip the videos themselves we can simply perform this clipping in a subsequent stage without the need for re-computing those cuts. We would do this by *just* passing in the CutDetectionSubsampler specification without the *args* but adding `cuts_are_clips`. video2dataset would then just extract the cut information from the metadata and use that for clipping. It would look something like this:
```yaml
...
    CutDetectionSubsampler:
        cuts_are_clips: True
    ClippingSubsampler:
        args:
            min_length: 4.0
            max_length: 20.0
            max_length_strategy: "all"
            precision: "keyframe_adjusted"
...
```




### Reading

### Storage

### Distribution




This module exposes a single function `download` which takes the same arguments as the command line tool:

* **url_list** A file with the list of url of images to download. It can be a folder of such files. (*required*)
* **output_folder** The path to the output folder. (default *"images"*)
* **processes_count** The number of processes used for downloading the pictures. This is important to be high for performance. (default *1*)
* **encode_formats** Dict of (modality, format) pairs specifying what file format each modality should be saved as. This determines which modalities will be written in the output dataset f.e. if we only specify audio only audio wil be saved (default *{"video": "mp4"}*)
* **output_format** decides how to save pictures (default *files*)
  * **files** saves as a set of subfolder containing pictures
  * **webdataset** saves as tars containing pictures
  * **parquet** saves as parquet containing pictures as bytes
  * **tfrecord** saves as tfrecord containing pictures as bytes
  * **dummy** does not save. Useful for benchmarks
* **input_format** decides how to load the urls (default *txt*)
  * **txt** loads the urls as a text file of url, one per line
  * **csv** loads the urls and optional caption as a csv
  * **tsv** loads the urls and optional caption as a tsv
  * **tsv.gz** loads the urls and optional caption as a compressed (gzip) tsv.gz
  * **json** loads the urls and optional caption as a json
  * **parquet** loads the urls and optional caption as a parquet
* **url_col** the name of the url column for parquet and csv (default *url*)
* **caption_col** the name of the caption column for parquet and csv (default *None*)
* **clip_col** the name of the column with a list of timespans for each clip (defualt *None*)
* **save_additional_columns** list of additional columns to take from the csv/parquet files and save in metadata files (default *None*)
* **number_sample_per_shard** the number of sample that will be downloaded in one shard (default *10000*)
* **timeout** maximum time (in seconds) to wait when trying to download an image (default *10*)
* **video_size** size of video frames (default *360*)
* **resize_mode** what resizing transformations to apply to video resolution (default *None*)
  * **scale** scale video keeping aspect ratios (currently always picks video height)
  * **crop** center crop to video_size x video_size
  * **pad** center pad to video_size x video_size
* **video_fps** what FPS to resample the video to. If < 0 then video FPS remains unchanged (default *-1*)
* **audio_rate** audio sampling rate, by default (-1) it is left unchanged from the downloaded video (default *-1*)
* **enable_wandb** whether to enable wandb logging (default *False*)
* **wandb_project** name of W&B project used (default *video2dataset*)
* **oom_shard_count** the order of magnitude of the number of shards, used only to decide what zero padding to use to name the shard files (default *5*)
* **distributor** choose how to distribute the downloading (default *multiprocessing*)
  * **multiprocessing** use a multiprocessing pool to spawn processes
  * **pyspark** use a pyspark session to create workers on a spark cluster (see details below)
* **subjob_size** the number of shards to download in each subjob supporting it, a subjob can be a pyspark job for example (default *1000*)
* **incremental_mode** Can be "incremental" or "overwrite". For "incremental", video2dataset will download all the shards that were not downloaded, for "overwrite" video2dataset will delete recursively the output folder then start from zero (default *incremental*)
* **tmp_dir** name of temporary directory in your file system (default */tmp*)
* **yt_metadata_args** dict of YouTube metadata arguments (default *None*, more info below)
* **detect_cuts** whether or not to detect jump-cuts in each video and store as metadata (default *False*)
* **cut_detection_mode** Can be either "longest" or "all" -- "longest" will select the longest contiguous (i.e. no jump-cuts) section of video, and "all" will select all contiguous sections of video to store in metadata (default *"longest"*)
* **cut_framerates** a list of additional framerates to detect jump cuts at. If None, jump cuts will only be detected at the original framerate of the video (default *None*)
* **cuts_are_clips** whether or not to turn each contiguous section of each input video into a distinct ouput video (default *False*)
* **cut_detector_threshold** mean pixel difference to trigger a jump cut detection for the cut detector. A lower threshold yields a more sensitive cut detector with more jump cuts. (default *27*) 
* **cut_detector_min_scene_len** minimum scene length for the cut detector (in frames). If the detector detects a jump cut and the distance from the previous cut is less than *cut_detector_min_scene_len* then the jump cut will not be annotated. (default *15*)
* **stage** which stage of processing to execute in betweeen downloading + cheap subsampling and costly subsampling (default *"download"*)
* **optical_flow_params** Dict containing parameters for optical flow detection. Keys can include *detector* ("cv2" or "raft", default "cv2"), *fps* (-1 or target fps, default *-1*), *detector_args* (additional args to pass to optical flow detector, default *None*), *downsample_size* (size to downsample shortest side of video to when detecting optical flow, default *None*), and *dtype* (datatype to store optical flow data in, default *np.float16*). Optional keys: *device* (only when using RAFT detector, lets you specify the device for the RAFT detector).
* **min_clip_length** Minimum length in seconds of a clip. Below this the subsampler will reject the clips (default *0.0*)
* **max_clip_length** Maximum length in seconds of a clip. Above this the ClippingSubsampler cuts up the long clip according to the max_clip_length_strategy (default *999999.0*)
* **max_clip_length_strategy** Tells the ClippingSubsampler how to resolve clips that are too long. "first" means take the first max_clip_length clip, "all" means take all contiguous max_clip_length clips (default *"all"*)
* **clipping_precision** informs the ClippingSubsampler how to clip the videos (see docstring for options (default *"low"*)
* **extract_compression_metadata** If True, extracts information about codec and puts in metadata (default *False*)


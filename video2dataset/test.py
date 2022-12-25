from data_loader import WebdatasetReader

TARS = "pipe: aws s3 cp s3://s-laion/acav100m/mp4_acav100m/{00000..00010}.tar -"

ds = WebdatasetReader(
    sampler=lambda a: a,
    preprocess= lambda a: a,
    input_dataset=TARS,
    batch_size=1,
    num_prepro_workers=12,
    enable_metadata=True,
)

for b in ds:
    print(b["video_tensor"].shape)






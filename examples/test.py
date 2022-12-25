import time
from data_loader import WebdatasetReader

TARS = "pipe: aws s3 cp s3://s-laion/acav100m/mp4_acav100m/{00000..00100}.tar -"

BS = 64
ds = WebdatasetReader(
    sampler=lambda a: a,
    preprocess= lambda a: a[0, :10, :10, :],
    input_dataset=TARS,
    batch_size=BS,
    num_prepro_workers=96,
    enable_metadata=True,
)

ct = 0

t0 = time.time()
for b in ds:
    ct += 1
    print(ct)
    print(b["video_tensor"].shape)
    if ct > 200:
        break
tf = time.time()

print("VID/S:")
print(ct*BS/(tf-t0))

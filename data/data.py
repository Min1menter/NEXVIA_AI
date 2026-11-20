from monai.apps import download_and_extract

download_and_extract(
    url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
    filepath="data/raw/msd/Task09_Spleen.tar",
    output_dir="data/raw/msd",
    hash_val="410d4a301da4e5b2f6f86ec3ddba524e",
    hash_type="md5",
)
import boto3

import datetime
import os
import hashlib
from typing import Dict

LOCAL_CACHE_PATH = "~/.points"


def filename_to_id(filename: str) -> int:
    """
    Hash filename and modification time to create a unique id.
    """
    mod_time = os.path.getmtime(filename)
    unique_string = filename + str(mod_time)
    hash_object = hashlib.md5(unique_string.encode())
    return int(hash_object.hexdigest()[:8], 16)


def filter_coco_images(coco_dict: Dict) -> Dict:
    # Filter out images that have no annotations.
    filtered_coco_dict = {
        "images": [],
        "annotations": [],
        "categories": coco_dict["categories"],
    }
    image_ids = set(anno["image_id"] for anno in coco_dict["annotations"])
    for image in coco_dict["images"]:
        if image["id"] in image_ids:
            filtered_coco_dict["images"].append(image)
    for anno in coco_dict["annotations"]:
        if anno["image_id"] in image_ids:
            filtered_coco_dict["annotations"].append(anno)
    print(
        f'Filtering {len(coco_dict["images"]) - len(filtered_coco_dict["images"])} out of {len(coco_dict["images"])} images without annotations'
    )
    return filtered_coco_dict


def get_s3_creds():
    access_key_id = os.environ.get("AWS_ACCESS_KEY_ID", None)
    secret_key_id = os.environ.get("AWS_SECRET_ACCESS_KEY", None)

    if access_key_id is None or secret_key_id is None:
        raise RuntimeError(
            "AWS credentials not found in environment variables."
            "Please set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in environment variables."
        )

    url = os.environ.get("AWS_S3_ENDPOINT_URL", None)
    if url is None:
        raise RuntimeError(
            "AWS S3 endpoint URL not found in environment variables."
            "Please set AWS_S3_ENDPOINT_URL in environment variables."
        )
    return access_key_id, secret_key_id, url


def build_s3_client() -> boto3.client:
    access_key_id, secret_key_id, url = get_s3_creds()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_key_id,
        endpoint_url=url,
    )
    return s3_client


def upload_file_to_s3(local_file_path, bucket_name, s3_folder):
    s3 = build_s3_client()
    s3_key = s3_folder + "/" + local_file_path.split("/")[-1]
    s3.upload_file(local_file_path, bucket_name, s3_key)
    print(f"Uploaded {local_file_path} to {bucket_name}/{s3_key}")


def ensure_local_cache_and_file(cache_dir, filename, bucket_name, s3_key):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory: {cache_dir}")

    local_file_path = os.path.join(cache_dir, filename)

    if not os.path.exists(local_file_path):
        s3 = build_s3_client()
        s3_key = os.path.join(s3_key, filename)
        s3.download_file(bucket_name, s3_key, local_file_path)
        print(f"Downloaded {s3_key} from {bucket_name} to {local_file_path}")

    else:
        timestamp = os.path.getmtime(local_file_path)
        last_modified_date = datetime.datetime.fromtimestamp(timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(
            f"File {filename} already exists in cache. Last modified on: {last_modified_date}"
        )


def list_contents_in_s3_directory(bucket_name, directory_prefix):
    s3 = build_s3_client()

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket_name, Prefix=directory_prefix, Delimiter="/"
    ):
        for prefix in page.get("CommonPrefixes", []):
            print("DIRECTORY:", prefix["Prefix"])

        for content in page.get("Contents", []):
            print("FILE:", content["Key"])


def create_directory_in_s3(bucket_name, directory_name):
    """
    Create a directory (or prefix) within an S3 bucket.

    Parameters:
    - bucket_name: The name of the S3 bucket.
    - directory_name: The directory (or prefix) to create.
                      This function will ensure it ends with a '/'.
    """
    if not directory_name.endswith("/"):
        directory_name += "/"

    s3 = build_s3_client()

    s3.put_object(Bucket=bucket_name, Key=directory_name, Body=b"")

    print(f"Created directory {directory_name} in bucket {bucket_name}")

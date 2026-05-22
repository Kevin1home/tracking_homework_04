"""Утилиты для работы с MinIO/S3."""
import boto3
from botocore.client import Config


def get_s3_client(endpoint_url: str, access_key: str, secret_key: str, region: str = "us-east-1"):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name=region,
    )


def ensure_bucket(client, bucket: str):
    existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if bucket not in existing:
        client.create_bucket(Bucket=bucket)


def upload_file(client, local_path: str, bucket: str, key: str):
    ensure_bucket(client, bucket)
    client.upload_file(local_path, bucket, key)


def download_file(client, bucket: str, key: str, local_path: str):
    client.download_file(bucket, key, local_path)

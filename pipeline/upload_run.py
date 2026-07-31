"""Upload a run folder to S3-compatible Object Storage: python -m pipeline.upload_run <run_dir>.

Nebius Object Storage speaks the S3 protocol; credentials are Nebius-issued
keys passed explicitly from S3_* variables (no AWS involvement, and boto3 is
just the standard S3 protocol client). Without S3_BUCKET the step is a
documented no-op so the pipeline also works in local-only setups.
"""

import os
import sys
from pathlib import Path


def main(run_dir: str) -> None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        print("S3_BUCKET not set; skipping artifact upload")
        return

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
    )
    run_path = Path(run_dir)
    prefix = f"runs/{run_path.name}"
    files = [p for p in sorted(run_path.rglob("*")) if p.is_file()]
    for path in files:
        key = f"{prefix}/{path.relative_to(run_path)}"
        client.upload_file(str(path), bucket, key)
    print(f"uploaded {len(files)} files to s3://{bucket}/{prefix}")


if __name__ == "__main__":
    main(sys.argv[1])

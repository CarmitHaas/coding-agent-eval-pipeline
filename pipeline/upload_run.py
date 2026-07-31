"""Upload a run folder to S3-compatible Object Storage: python -m pipeline.upload_run <run_dir>.

Reads S3_BUCKET and S3_ENDPOINT_URL (plus standard AWS_* credentials) from the
environment. Without S3_BUCKET the step is a documented no-op so the pipeline
also works in local-only setups.
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

    client = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"))
    run_path = Path(run_dir)
    prefix = f"runs/{run_path.name}"
    files = [p for p in sorted(run_path.rglob("*")) if p.is_file()]
    for path in files:
        key = f"{prefix}/{path.relative_to(run_path)}"
        client.upload_file(str(path), bucket, key)
    print(f"uploaded {len(files)} files to s3://{bucket}/{prefix}")


if __name__ == "__main__":
    main(sys.argv[1])

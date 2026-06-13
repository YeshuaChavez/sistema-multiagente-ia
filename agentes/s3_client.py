# -*- coding: utf-8 -*-
"""
Utilidades S3 compartidas por todos los agentes del SMA-ML/DL.
Bucket: epipredict-dengue
"""

import os
import boto3
from botocore.exceptions import ClientError

BUCKET            = "epipredict-dengue"
PREFIX_CRUDOS     = "datos_crudos/"
PREFIX_PROCESADOS = "datos_procesados/"
PREFIX_MODELOS    = "modelos/"


def _client():
    return boto3.client("s3")


def exists(s3_key: str) -> bool:
    try:
        _client().head_object(Bucket=BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False


def download(s3_key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    _client().download_file(BUCKET, s3_key, local_path)
    print(f"   [S3↓] {s3_key}")


def upload(local_path: str, s3_key: str) -> None:
    _client().upload_file(local_path, BUCKET, s3_key)
    print(f"   [S3↑] {s3_key}")


def download_prefix(s3_prefix: str, local_dir: str) -> None:
    """Descarga todos los objetos bajo s3_prefix a local_dir."""
    paginator = _client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(s3_prefix):]
            if not rel:
                continue
            download(key, os.path.join(local_dir, rel))


def upload_dir(local_dir: str, s3_prefix: str) -> None:
    """Sube todos los archivos de local_dir a s3_prefix."""
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir).replace("\\", "/")
            upload(local_path, s3_prefix + rel)


def ensure_local(s3_key: str, local_path: str) -> bool:
    """Descarga s3_key a local_path solo si no existe localmente. Retorna True si disponible."""
    if os.path.exists(local_path):
        return True
    try:
        download(s3_key, local_path)
        return True
    except Exception as e:
        print(f"   [S3] No se pudo descargar {s3_key}: {e}")
        return False

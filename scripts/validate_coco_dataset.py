#!/usr/bin/env python3
"""Strict structural and geometric validation for dataset_v2 COCO files."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset_v2"


def validate_split(split: str) -> list[str]:
    errors: list[str] = []
    split_dir = DATASET / split
    image_dir = split_dir / "images"
    annotation_path = split_dir / "annotations.json"
    if not annotation_path.is_file():
        return [f"{split}: missing {annotation_path}"]
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))

    categories = payload.get("categories", [])
    if categories != [{"id": 1, "name": "bag", "supercategory": "object"}]:
        errors.append(f"{split}: categories must contain only id=1 name=bag")

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    image_ids = [item.get("id") for item in images]
    annotation_ids = [item.get("id") for item in annotations]
    if len(image_ids) != len(set(image_ids)):
        errors.append(f"{split}: duplicate image ids")
    if len(annotation_ids) != len(set(annotation_ids)):
        errors.append(f"{split}: duplicate annotation ids")

    by_id = {}
    referenced_names = set()
    for item in images:
        image_id = item.get("id")
        file_name = item.get("file_name")
        path = image_dir / str(file_name)
        if not path.is_file():
            errors.append(f"{split}: missing image {file_name}")
            continue
        image = cv2.imread(str(path))
        if image is None:
            errors.append(f"{split}: unreadable image {file_name}")
            continue
        height, width = image.shape[:2]
        if item.get("width") != width or item.get("height") != height:
            errors.append(f"{split}: wrong dimensions for {file_name}")
        by_id[image_id] = (width, height, file_name)
        referenced_names.add(file_name)

    actual_names = {path.name for path in image_dir.glob("*") if path.is_file()}
    for extra in sorted(actual_names - referenced_names):
        errors.append(f"{split}: unreferenced image file {extra}")

    for item in annotations:
        prefix = f"{split}: annotation {item.get('id')}"
        if item.get("category_id") != 1:
            errors.append(f"{prefix}: category_id is not 1")
        image_id = item.get("image_id")
        if image_id not in by_id:
            errors.append(f"{prefix}: unknown image_id {image_id}")
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{prefix}: bbox must be [x,y,width,height]")
            continue
        x, y, w, h = bbox
        width, height, _ = by_id[image_id]
        if not all(isinstance(value, (int, float)) for value in bbox):
            errors.append(f"{prefix}: bbox values must be numeric")
            continue
        if w <= 0 or h <= 0:
            errors.append(f"{prefix}: width and height must be positive")
        if x < 0 or y < 0 or x + w > width or y + h > height:
            errors.append(f"{prefix}: bbox outside image bounds")
        if item.get("area") != w * h:
            errors.append(f"{prefix}: area does not equal width*height")
        if item.get("iscrowd") != 0:
            errors.append(f"{prefix}: iscrowd must be 0")
    return errors


def main() -> None:
    all_errors = validate_split("train") + validate_split("val")
    if all_errors:
        print("COCO validation FAILED")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("COCO validation PASSED for train and val")


if __name__ == "__main__":
    main()

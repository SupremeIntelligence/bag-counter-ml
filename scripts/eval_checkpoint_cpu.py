import argparse
import json
from pathlib import Path

from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


CONFIG_PATH = "configs/rtmdet_bag.py"
VAL_IMAGES_DIR = Path("data/dataset_v2/val/images")
VAL_ANNOTATIONS = Path("data/dataset_v2/val/annotations.json")


def evaluate(checkpoint_path: str):
    print(f"\nEvaluating: {checkpoint_path}")
    print("Device: CPU")

    model = init_detector(
        CONFIG_PATH,
        checkpoint_path,
        device="cpu",
    )

    coco_gt = COCO(str(VAL_ANNOTATIONS))

    predictions = []

    image_ids = coco_gt.getImgIds()

    for index, image_id in enumerate(image_ids, start=1):
        image_info = coco_gt.loadImgs(image_id)[0]

        image_path = VAL_IMAGES_DIR / image_info["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        result = inference_detector(model, str(image_path))

        instances = result.pred_instances.cpu()

        bboxes = instances.bboxes.numpy()
        scores = instances.scores.numpy()
        labels = instances.labels.numpy()

        for bbox, score, label in zip(
            bboxes,
            scores,
            labels,
        ):
            x1, y1, x2, y2 = bbox.tolist()

            width = x2 - x1
            height = y2 - y1

            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": 1,

                    #COCO формат
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(width),
                        float(height),
                    ],

                    "score": float(score),
                }
            )

        print(f"\rProcessed {index}/{len(image_ids)} images", end="", flush=True)

    print()

    if not predictions:
        print("Model produced no detections.")
        return

    output_path = Path(
        "work_dirs/rtmdet_bag/"
        + Path(checkpoint_path).stem
        + "_val_predictions.json"
    )

    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Predictions saved to: {output_path}")

    coco_dt = coco_gt.loadRes(str(output_path))

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    print("\nMain metrics:")
    print(f"bbox_mAP    = {evaluator.stats[0]:.4f}")
    print(f"bbox_mAP_50 = {evaluator.stats[1]:.4f}")
    print(f"bbox_mAP_75 = {evaluator.stats[2]:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint", help="Path to RTMDet checkpoint")

    args = parser.parse_args()

    evaluate(args.checkpoint)


if __name__ == "__main__":
    main()
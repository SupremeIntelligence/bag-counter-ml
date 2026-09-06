from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv


config_file = "rtmdet_tiny_8xb32-300e_coco.py"
checkpoint_file = ("rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth")

image_path = "test.png"

model = init_detector(config_file, checkpoint_file, device="cpu")

result = inference_detector(model, image_path)

image = mmcv.imread(image_path)
image = mmcv.imconvert(image, "bgr", "rgb")
visualizer = DetLocalVisualizer()
visualizer.dataset_meta = model.dataset_meta

visualizer.add_datasample(
    name="result",
    image=image,
    data_sample=result,
    draw_gt=False,
    show=False,
    pred_score_thr=0.2,
    out_file="result.png",
)

print("Saved result to result.png")
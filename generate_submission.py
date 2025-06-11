# generate_submission.py

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_yaml,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

def save_one_json(predn, jdict, path, index, class_map):
    """
    Saves one JSON detection result with image ID, category ID, bounding box, and score.
    """
    # image_id를 다시 숫자 인덱스로 사용하도록 복구합니다.
    image_name = path.stem
    for p, b in zip(predn.tolist(), box.tolist()):
        if p[4] < 0.1:
            continue
        jdict.append(
            {
                "image_id": int(index),      # ✅ 숫자로 된 고유 인덱스 사용
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )
        
@smart_inference_mode()
def run(
    data,
    weights=None,
    batch_size=16,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=300,
    device="",
    workers=8,
    single_cls=False,
    project=ROOT / "runs/submission",
    name="exp",
    exist_ok=False,
    half=True,
    dnn=False,
    rgbt=False,
):
    # 초기 설정
    device = select_device(device, batch_size=batch_size)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # runs/submission/exp1
    save_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    half &= pt and device.type != "cpu"  # half precision
    
    # 데이터 로드
    data = check_dataset(data)  # check
    
    # Warmup
    if rgbt:
        LOGGER.info('RGBT model warmup...')
        dummy_input_list = [
            torch.zeros(1, 3, imgsz, imgsz, device=model.device),
            torch.zeros(1, 3, imgsz, imgsz, device=model.device)
        ]
        if half:
            dummy_input_list = [d.half() for d in dummy_input_list]
        if pt:
            model.model(dummy_input_list)
    else:
        model.warmup(imgsz=(1, 3, imgsz, imgsz))

    dataloader = create_dataloader(
        data['test'],  # 'test' 데이터셋 경로 사용
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=0.5,
        rect=False, # Rectangular inference
        workers=workers,
        prefix=colorstr("test: "),
        rgbt_input=rgbt
    )[0]

    # 추론 시작
    model.eval()
    jdict = []
    seen = 0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)
    
    class_map = list(range(1000)) # 클래스 ID 매핑용

    pbar = tqdm(dataloader, desc="Generating submission file...", bar_format=TQDM_BAR_FORMAT)
    for batch_i, (ims, targets, paths, shapes, indices) in enumerate(pbar):
        with dt[0]:
            if rgbt:
                ims = [im.to(device, non_blocking=True).half() if half else im.to(device, non_blocking=True).float() / 255 for im in ims]
            else:
                ims = ims.to(device, non_blocking=True)
                ims = ims.half() if half else ims.float() / 255
        
        # Inference
        with dt[1]:
            preds = model(ims, augment=False)

        # NMS
        with dt[2]:
            preds = non_max_suppression(preds, conf_thres, iou_thres, classes=None, agnostic=single_cls, max_det=max_det)

        # 결과 처리
        for si, pred in enumerate(preds):
            seen += 1
            path, shape = Path(paths[si]), shapes[si][0]
            index = indices[si] # 각 이미지의 고유 인덱스 (0, 1, 2...)
            
            if len(pred) == 0:
                continue

            # 원본 이미지 크기에 맞게 예측 좌표 스케일링
            predn = pred.clone()
            scale_boxes(ims[0].shape[1:], predn[:, :4], shape, shapes[si][1])  
            
            # JSON 형식으로 저장
            save_one_json(predn, jdict, path, index, class_map)

    # JSON 파일 저장
    w = Path(weights[0] if isinstance(weights, list) else weights).stem
    pred_json_path = str(save_dir / f"{w}_submission.json")
    LOGGER.info(f"\nSaving submission file to {pred_json_path}...")
    with open(pred_json_path, "w") as f:
        json.dump(jdict, f)
        
    LOGGER.info(f"Submission file generation complete. You can now submit '{pred_json_path}' to eval.ai.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/kaist-rgbt.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", type=str, required=True, help="model.pt path")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--project", default=ROOT / "runs/submission", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--rgbt", action="store_true", help="Feed RGB-T multispectral image pair.")
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
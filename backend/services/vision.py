"""이미지 분류(ResNet-18) 및 객체 탐지(YOLOv8n)"""
from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=1)
def _resnet():
    import torchvision
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights).eval()
    return model, weights


def classify(img: Image.Image, top_k: int = 3) -> dict:
    import torch
    model, weights = _resnet()
    x = weights.transforms()(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
    vals, idxs = prob.topk(top_k)
    cats = weights.meta["categories"]
    top = [{"label": cats[int(i)], "prob": round(float(v), 4)} for v, i in zip(vals, idxs)]
    first = int(idxs[0])
    # ImageNet 인덱스: 151~268 개 품종, 281~285 고양이류
    group = "DOG" if 151 <= first <= 268 else "CAT" if 281 <= first <= 285 else "OTHER"
    return {"top": top, "group": group}


@lru_cache(maxsize=1)
def _yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def detect(img: Image.Image, conf: float = 0.25) -> tuple[list[dict], Image.Image]:
    result = _yolo().predict(img.convert("RGB"), conf=conf, verbose=False)[0]
    dets = [{
        "label": result.names[int(b.cls)],
        "conf": round(float(b.conf), 3),
        "box": [round(float(v), 1) for v in b.xyxy[0]],
    } for b in result.boxes]
    annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR -> RGB
    return dets, annotated

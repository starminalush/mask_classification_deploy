import os
from io import BytesIO
from typing import List, Tuple

import aiohttp.web
import cv2 as cv
import pytest
import mlflow
import mxnet as mx
import numpy as np
import torch
from PIL import Image
from app.modules.mtcnn_detector import MtcnnDetector
from dotenv import load_dotenv
from loguru import logger
from torchvision import transforms

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
routes = aiohttp.web.RouteTableDef()
detector: MtcnnDetector = MtcnnDetector(
    model_folder="/app/app/modules/mtcnn_model",
    ctx=mx.cpu(),
    num_worker=1,
    accurate_landmark=True,
    threshold=[0.7, 0.8, 0.9],
)

model = mlflow.pytorch.load_model(
    os.getenv("MODEL_PATH"), map_location=torch.device("cpu")
)
model.eval()


def bytes_encode_image(image: np.ndarray) -> bytes:
    return BytesIO(cv.imencode(".jpg", image)[1].tostring()).read()


def preprocess_image(img: bytes) -> np.ndarray:
    image: np.ndarray = np.asarray(bytearray(img), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image


transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def prepare_image(image: np.ndarray) -> torch.Tensor:
    image: Image = Image.fromarray(image)
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor




@routes.post("/detect_image")
async def detect_image(request: aiohttp.web.Request):
    """
    метод, который получает картинку, находит на ней лица,
    определяет, если ли на каждом нвйденном лице маска и
    возвращает в ответ картинку с нарисованными bbox
    и метками класса возле каждого лица
    :param request: входные данные http запроса
    :type request: aiohttp.web.Request
    :return: image if success, else status code >= 400
    :rtype:
    """
    reader = await request.multipart()

    image = await reader.next()
    if not image:
        return aiohttp.web.HTTPBadRequest(reason="Please provide image file.")
    if image:
        img_content: bytes = await image.read()
    else:
        return {"error:wrong image"}, 400

    im: np.ndarray = preprocess_image(img_content)

    detection_result: List = detector.detect_face(im)
    for i in detection_result[0]:
        bbox: List = i
        bbox = [int(i) for i in bbox]
        bbox = [0 if i < 0 else i for i in bbox]
        croped_frame: np.ndarray = im[bbox[1] : bbox[3], bbox[0] : bbox[2]]

        image_tensor: torch.Tensor = prepare_image(croped_frame)
        result: torch.Tensor = model(image_tensor)
        logger.info(result)
        _, predicted = torch.max(result.data, 1)
        result: int = predicted.item()
        color: Tuple = ()
        text: str = ""
        if result == 1:
            color = (0, 0, 255)
            text = "without_mask"
        if result == 0:
            color = (0, 255, 0)
            text = "with_mask"
        cv.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 5)
        cv.putText(
            im, text, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
    result_image: bytes = bytes_encode_image(im)
    return aiohttp.web.Response(body=result_image, content_type="image/jpeg")

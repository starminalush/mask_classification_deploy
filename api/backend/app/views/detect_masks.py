import os
from io import BytesIO

import aiohttp.web
import cv2 as cv
import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from mtcnn import MTCNN
from torchvision import transforms

load_dotenv()

os.environ['MLFLOW_TRACKING_URI'] = os.getenv("MLFLOW_TRACKING_URI")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
routes = aiohttp.web.RouteTableDef()
detector = MTCNN()

model = mlflow.pytorch.load_model(os.getenv('MODEL_PATH'), map_location=torch.device('cpu'))
model.eval()


def bytes_encode_image(image):
    return BytesIO(cv.imencode('.jpg', image)[1].tostring()).read()


def preprocess_image(img: bytes):
    image = np.asarray(bytearray(img), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image


transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


def prepare_image(image):
    image = Image.fromarray(image)
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


@routes.post('/detect_image')
async def detect_image(request: aiohttp.web.Request):
    reader = await request.multipart()

    image = await reader.next()
    if not image:
        return aiohttp.web.HTTPBadRequest(reason='Please provide image file.')
    if image:
        img_content = await  image.read()
    else:
        return {'error:wrong image'}, 400

    im = preprocess_image(img_content)

    detection_result = detector.detect_faces(im)
    for i in detection_result:
        bbox = i['box']
        croped_frame = im[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]

        image_tensor = prepare_image(croped_frame)
        result = model(image_tensor)
        logger.info(result)
        _, predicted = torch.max(result.data, 1)
        result = predicted.item()
        color = None
        if result == 1:
            color = (0, 0, 255)
        if result == 0:
            color = (0, 255, 0)
        cv.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 5)
    result = bytes_encode_image(im)
    return aiohttp.web.Response(body=result, content_type="image/jpeg")

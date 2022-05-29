import aiohttp.web
import mlflow
import numpy as np
import cv2
from PIL import Image
import io
from loguru import logger
from torchvision import transforms
import os
import torch


os.environ['MLFLOW_TRACKING_URI'] = "http://mlflow_server:5000"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = "s3keys3key"
os.environ['AWS_SECRET_ACCESS_KEY'] = "s3keys3key"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
routes = aiohttp.web.RouteTableDef()




model = mlflow.pytorch.load_model('runs:/26e63663f1424cc7a8e9049e53ddb5de/resnet', map_location=torch.device('cpu'))
model.eval()
transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


def prepare_image(image):
    image = Image.open(io.BytesIO(image))
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


@routes.post('/detect')
async def detect(request: aiohttp.web.Request):
    logger.info('here')
    post = await request.post()
    logger.info(post)
    image = post.get('image')
    logger.info(image)
    if image:
        img_content = image.file.read()  # type: ignore
    else:
        return {'error:wrong image'}, 400
    image_tensor = prepare_image(img_content)
    result = model(image_tensor)
    _, predicted = torch.max(result.data, 1)
    result = predicted.item()
    return aiohttp.web.json_response(
        {
            'class': result
        }
    )

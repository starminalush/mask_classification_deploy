import io
import os

import aiohttp.web
import mlflow
import numpy as np
import torch
from PIL import Image, ImageDraw
from loguru import logger
from mtcnn import MTCNN
from torchvision import transforms

os.environ['MLFLOW_TRACKING_URI'] = "http://mlflow_server:5000"
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = "s3keys3key"
os.environ['AWS_SECRET_ACCESS_KEY'] = "s3keys3key"
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
routes = aiohttp.web.RouteTableDef()
detector = MTCNN()

model_class = {
    0: 'with_mask',
    1: 'without_mask'
}
model = mlflow.pytorch.load_model('runs:/6009aae23d044b3ab9302e4283f8d0f1/mobilenet', map_location=torch.device('cpu'))
model.eval()
transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)


def prepare_image(image):
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


@routes.post('/detect')
async def detect(request: aiohttp.web.Request):
    logger.info('here')
    post = await request.post()
    logger.info(post)
    image = post.get('image')
    if image:
        img_content = image.file.read()  # type: ignore
    else:
        return {'error:wrong image'}, 400

    im = Image.open(io.BytesIO(img_content))

    detection_result = detector.detect_faces(np.asarray(im))
    for i in detection_result:
        bbox = i['box']
        croped_frame = im.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        croped_frame.save("test.jpg")

        image_tensor = prepare_image(croped_frame)
        result = model(image_tensor)
        _, predicted = torch.max(result.data, 1)
        result = predicted.item()
        crop_class = model_class.get(result)
        img_draw = ImageDraw.Draw(im)

        if result == 1:
            outline_color = 'Red'
        else:
            outline_color = 'Green'
        img_draw.rectangle(((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])), outline=outline_color)
        img_draw.text((bbox[0], bbox[1]), crop_class, fill='green')

    stream = io.BytesIO()
    im.save(stream, "JPEG")

    return aiohttp.web.Response(body=stream.getvalue(), content_type="image/png")

# mask_classification_deploy

Сервисы для деплоя обученной модели классификации масок

**Frontend**

![image](https://user-images.githubusercontent.com/103132748/173671969-a8c448be-d416-4131-9d9e-3a8ae8ceb244.png)


**Запуск сервисов**
1. Заполнить файл api/build/.env, указав в нем следующие переменные

MLFLOW_TRACKING_URI=mlflow tracking uri

MLFLOW_S3_ENDPOINT_URL=mlflow s3 endpoint url

AWS_ACCESS_KEY_ID=aws access key

AWS_SECRET_ACCESS_KEY=aws secret key

MODEL_PATH=model path from mlflow

2. docker-compose up --build -d.
3. Profit :)

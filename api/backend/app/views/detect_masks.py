import aiohttp.web
from aiohttp_jinja2 import template


routes = aiohttp.web.RouteTableDef()

@routes.get('/detect')
@template("detect.html")
async def detect(request: aiohttp.web.Request):
    '''
    http метод для обнаружения лиц
    :param request:
    :return:
    '''

    pass
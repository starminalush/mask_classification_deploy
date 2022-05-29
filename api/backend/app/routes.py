from aiohttp.web import Application

from .views import detect_masks


def setup_routes(app: Application):
    app.add_routes(detect_masks.routes)

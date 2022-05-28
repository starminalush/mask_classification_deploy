from aiohttp.web import Application
from aiohttp_basicauth_middleware import basic_auth_middleware

from .views import detect_masks


def setup_routes(app: Application):
    app.add_routes(detect_masks.routes)
    app.router.add_static('/static', path='app/static')

    app.middlewares.append(basic_auth_middleware(('/demo',), {'test': 'test'}))

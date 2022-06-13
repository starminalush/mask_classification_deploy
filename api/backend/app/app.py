import aiohttp_cors
from aiohttp import web, ClientSession

from .routes import setup_routes


async def startup(app: web.Application):
    app["http_session"] = ClientSession()


async def cleanup(app: web.Application):
    await app["http_session"].close()


async def create_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    setup_routes(app)
    return app


app = create_app()

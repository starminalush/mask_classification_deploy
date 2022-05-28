import aiohttp_cors
import aiohttp_jinja2
import jinja2
from aiohttp import web, ClientSession

from .routes import setup_routes


async def startup(app: web.Application):
    app['http_session'] = ClientSession()


async def cleanup(app: web.Application):
    await app['http_session'].close()


async def create_app() -> web.Application:
    app = web.Application(client_max_size=30 * 1024 ** 2)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    setup_routes(app)
    aiohttp_jinja2.setup(app, loader=jinja2.PackageLoader('app', 'templates'))

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)

    return app


app = create_app()

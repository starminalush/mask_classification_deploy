import aiohttp.web

from app import app

if __name__ == '__main__':
    aiohttp.web.run_app(app)

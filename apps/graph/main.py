import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

EE_API_URL = os.getenv("EE_API_URL")


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request) -> HTMLResponse:
    """
    Render the index page.
    """
    return templates.TemplateResponse(
        "index.html", {"request": request, "ee_api_url": EE_API_URL}
    )

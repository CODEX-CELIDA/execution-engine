import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Read the graph data from the JSON file
with open("cy_graph.json") as f:
    cy_graph = json.load(f)


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request) -> HTMLResponse:
    """
    Get the index page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/graph", response_class=JSONResponse)
async def get_graph() -> JSONResponse:
    """
    Get the graph data.
    """
    return JSONResponse(content=cy_graph)

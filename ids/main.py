from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse

from service.blacklist_service import is_blacklisted
from service.classification_service import access_log_classification

app = FastAPI()


@app.post("/classification")
async def root(request: Request):
    return access_log_classification()


@app.get("/echo/{name}")
async def say_echo(request: Request, name: str):
    log_ip = request.headers.get('Log-Ip')
    if is_blacklisted(log_ip):
        raise HTTPException(403, detail='Ip has been blacklisted!')
    return RedirectResponse('http://localhost:3000/echo/'+name, status_code=303)

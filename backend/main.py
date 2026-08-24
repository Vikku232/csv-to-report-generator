# backend/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import gradio as gr
from backend.api.routes import health, report
from frontend.app import demo

app = FastAPI(
    title="CSV Report Generator API",
    description="Automated profiling, baseline ML modeling, and executive PDF generation engine.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    return JSONResponse(
        status_code=400,
        content={"detail": f"Could not generate the report from this data: {exc}"},
    )

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(report.router, prefix="/api", tags=["report"])

@app.get("/", include_in_schema=False)
async def frontend_redirect():
    return RedirectResponse(url="/app")

app = gr.mount_gradio_app(app, demo, path="/app")
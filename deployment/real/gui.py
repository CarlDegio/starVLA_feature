"""StarVLA manual GUI profile; reuses YAM session, cameras, and E-stop unchanged."""
import argparse
import os
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from deployment.real.gui_session import ManualEpisodeSession
from deployment.real.infer import RecordedClient, ROOT


class StepRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8002, ge=1, le=65535)
    prompt: str = ""
    collection_id: str = Field(default="real_train_v1", min_length=1)
    seed_namespace: str = Field(default="real_train_v1", min_length=1)
    max_joint_speed: float = Field(default=0.3, gt=0, allow_inf_nan=False)


class LabelRequest(BaseModel):
    episode_id: str
    decision: str


def install_manual_profile(app, cfg_loader, output):
    session = app.state.session
    manager = ManualEpisodeSession(session, output, cfg_loader)
    app.state.manual_episode = manager
    # The base app mounts static files last. Insert this profile's routes before
    # that mount, retaining the camera, preview, status and emergency endpoints.
    static = app.router.routes.pop()

    @app.middleware("http")
    async def guard(request, call_next):
        path = request.url.path
        always_block = {"/api/deploy/start", "/api/deploy/reset", "/api/deploy/stop", "/api/jobs"}
        active_block = {"/api/collect/prepare-teleop", "/api/collect/start-teleop",
                        "/api/collect/start-recording", "/api/session/reset",
                        "/api/maintenance/reset-can", "/api/maintenance/zero-gello",
                        "/api/cameras/reset", "/api/cameras/kill-holder"}
        if request.method == "POST" and (path in always_block or
                (manager.phase != "idle" and path in active_block)):
            return JSONResponse(status_code=409, content={"detail": "StarVLA 手动采集请使用单次执行/结束 episode；先标记当前 episode"})
        return await call_next(request)

    @app.get("/")
    def page():
        return FileResponse(Path(__file__).with_name("gui.html"))

    @app.get("/api/real/status")
    def status():
        cfg = cfg_loader()
        result = manager.status()
        result.update(home_pose=manager.home, home_error=None, mock=session.mock,
                      cameras=[{"name": c.name, "role": c.role} for c in cfg.cameras],
                      output=str(output), estopped=session._estopped)
        return result

    @app.post("/api/real/check")
    def check(body: StepRequest):
        client = None
        try:
            client = RecordedClient(body.host, body.port, 10)
            return {"metadata": client.metadata, "motion_enabled": False}
        except Exception as exc:
            raise HTTPException(400, str(exc))
        finally:
            if client is not None:
                client.close()

    @app.post("/api/real/step")
    def step(body: StepRequest):
        try:
            return manager.step(**body.model_dump())
        except Exception as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/real/end")
    def end():
        try:
            return manager.end()
        except Exception as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/real/label")
    def label(body: LabelRequest):
        try:
            return manager.label(body.episode_id, body.decision)
        except Exception as exc:
            raise HTTPException(409, str(exc))

    app.router.routes.append(static)
    return app


def main():
    import uvicorn
    from yam_abc_reproduce.config import build_station_config
    from yam_abc_reproduce.gui.server import create_app
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="configs/station_yam.yaml")
    parser.add_argument("--cameras")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8042)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/LIBERO/edl_pred_real/records")
    args = parser.parse_args()
    os.environ.setdefault("YAM_ABC_VIDEO_ENCODER", "libx264")
    os.environ.setdefault("YAM_ABC_VIDEO_HWDECODE", "0")
    from yam_abc_reproduce.data import codec
    codec.encoder()
    def load():
        return build_station_config(args.station, args.cameras)
    app = create_app(load(), mock=args.mock, station_path=args.station, cameras_path=args.cameras)
    install_manual_profile(app, load, args.output.resolve())
    print(f"StarVLA manual GUI: http://{args.host}:{args.port} (hardware stays off until a user action)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()

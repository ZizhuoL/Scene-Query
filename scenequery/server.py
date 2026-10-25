"""Loopback application serving only explicit scene, image and static assets."""
import asyncio
import time
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .common import ROOT,DATA,ART,read_json
from .search import SceneSearch

engine=None
load_error=None
load_task=None


async def load():
    global engine,load_error
    try:engine=await asyncio.to_thread(SceneSearch)
    except Exception as error:
        load_error=str(error)


@asynccontextmanager
async def lifespan(app):
    global load_task
    load_task=asyncio.create_task(load())
    yield
    await load_task


app=FastAPI(title='SceneQuery',lifespan=lifespan)


class Query(BaseModel):
    text:str=Field(min_length=1,max_length=120)


@app.get('/api/status')
def status():
    return {'ready':engine is not None,'error':load_error}


@app.get('/api/scene')
def scene():
    frames=read_json(ROOT/'manifests/frames.json')['frames']
    train=read_json(ROOT/'reports/training.json')
    return {'name':'The Freiburg desk','dataset':'TUM RGB-D · fr1/desk','license':'CC BY 4.0','point_count':train['points'],'training_steps':train['config']['steps'],'training_seconds':round(train['seconds'],2),'frames':[{k:f[k] for k in ['id','c2w','split','timestamp','rgb']} for f in frames],'default_frame_id':42,'attribution':'Sturm et al., TUM RGB-D Benchmark, IROS 2012','method':'Locally optimized RGB-D surfels · frozen OWLv2 · multi-view association'}


@app.get('/api/points')
def points():
    fit=np.load(ART/'scene/fitted.npz')
    return Response(np.c_[fit['xyz'],fit['rgb']].astype('<f4').tobytes(),media_type='application/octet-stream')


def reconstruction_delivery():
    path=ROOT/'reports/reconstruction_delivery.json'
    if not path.exists():path=ROOT/'reports/neural/delivery.json'
    if not path.exists():raise HTTPException(404,'Neural reconstruction is not yet delivered')
    return read_json(path)


@app.get('/api/reconstruction')
def reconstruction():
    try:return {'available':True,**reconstruction_delivery()}
    except HTTPException:return {'available':False}


@app.get('/api/neural-points')
@app.get('/api/learned-points')
def neural_points():
    delivery=reconstruction_delivery()
    path=ROOT/delivery['surface_asset'] if 'surface_asset' in delivery else ART/'neural'/delivery['selected_run']/'surface.npz'
    if not path.resolve().is_relative_to(ART.resolve()):raise HTTPException(404,'Unknown surface asset')
    fit=np.load(path)
    return Response(np.c_[fit['xyz'],fit['rgb']].astype('<f4').tobytes(),media_type='application/octet-stream')


@app.get('/api/reconstruction/image/{run}/{frame_id}/{kind}')
def reconstruction_image(run:str,frame_id:int,kind:str):
    delivery=reconstruction_delivery()
    if run not in delivery['runs'] or frame_id not in delivery['final_ids'] or kind not in ('comparison','render'):
        raise HTTPException(404,'Unknown reconstruction evidence')
    name=f'frame_{frame_id:02}.jpg' if kind=='comparison' else f'render_{frame_id:02}.png'
    directory=delivery['runs'][run].get('artifact_directory',f'artifacts/neural/{run}/checkpoint_final')
    path=ROOT/directory/name
    if not path.resolve().is_relative_to(ART.resolve()) or not path.is_file():raise HTTPException(404,'Unknown reconstruction image')
    return FileResponse(path,media_type='image/jpeg' if kind=='comparison' else 'image/png')


@app.post('/api/query')
def query(q:Query):
    if engine is None:raise HTTPException(503,load_error or 'The scene index is loading.')
    try:return engine.query(q.text)
    except ValueError as error:raise HTTPException(422,str(error)) from error


@app.get('/api/photo/{frame_id}')
def photo(frame_id:int):
    frames=read_json(ROOT/'manifests/frames.json')['frames']
    if frame_id<0 or frame_id>=len(frames):raise HTTPException(404,'Unknown source frame')
    return FileResponse(DATA/frames[frame_id]['rgb'],media_type='image/png')


@app.get('/')
def index():return FileResponse(ROOT/'web/index.html')


app.mount('/web',StaticFiles(directory=ROOT/'web'),name='web')
app.mount('/vendor/three',StaticFiles(directory=ROOT/'node_modules/three'),name='three')

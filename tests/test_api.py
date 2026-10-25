"""Serving boundaries and input behavior; real search is tested separately."""
from fastapi.testclient import TestClient
from scenequery.server import app


def test_real_assets_and_input_validation():
    # Without lifespan the search remains unloaded; assets are still real.
    client=TestClient(app)
    assert client.get('/').status_code==200
    meta=client.get('/api/scene').json()
    raw=client.get('/api/points')
    assert raw.status_code==200
    assert len(raw.content)==meta['point_count']*6*4
    assert client.get('/api/photo/42').headers['content-type']=='image/png'
    assert client.get('/api/photo/-1').status_code==404
    assert client.get('/api/photo/9999').status_code==404
    assert client.post('/api/query',json={'text':''}).status_code==422
    assert client.post('/api/query',json={'text':'x'*121}).status_code==422
    assert client.get('/data/rgb.txt').status_code==404
    assert client.get('/.env').status_code==404


def test_neural_delivery_and_closed_asset_routes():
    client=TestClient(app)
    delivery=client.get('/api/reconstruction').json()
    assert delivery['available']
    raw=client.get('/api/neural-points')
    assert len(raw.content)==delivery['surface_points']*6*4
    run=delivery['selected_run'];frame=delivery['final_ids'][0]
    response=client.get(f'/api/reconstruction/image/{run}/{frame}/render')
    assert response.status_code==200 and response.headers['content-type']=='image/png'
    assert client.get(f'/api/reconstruction/image/unknown/{frame}/render').status_code==404
    assert client.get(f'/api/reconstruction/image/{run}/999/render').status_code==404
    assert client.get(f'/api/reconstruction/image/{run}/{frame}/checkpoint.pt').status_code==404

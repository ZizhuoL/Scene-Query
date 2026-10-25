"""Check cached query math against official forward on development query only."""
import numpy as np
import torch
from PIL import Image
from scenequery.common import ROOT,DATA,atomic_json
from scenequery.search import SceneSearch
from scenequery.detector import load_detector


def test_cached_logits_equal_official_forward():
    engine=SceneSearch(limit=1)
    _,model=load_detector()
    inputs=engine.processor(text=[engine.model_config['query_template'].format('monitor')],images=Image.open(DATA/engine.frames[0]['rgb']),return_tensors='pt')
    with torch.inference_mode():
        output=model(**inputs)
        reference=output.logits[0,:,0].numpy()
    query=engine.encode('monitor')
    cached=(engine.emb[0]@query+engine.shift[0])*engine.scale[0]
    difference=float(abs(reference-cached).max())
    atomic_json(ROOT/f"reports/detector_equivalence_{engine.model_config['family']}.json",{'development_query':'monitor','max_logit_difference':difference,'expected_tolerance':1e-4,'pass':difference<1e-4})
    np.testing.assert_allclose(cached,reference,atol=1e-4,rtol=1e-5)
    official_boxes=engine.processor.post_process_object_detection(output,threshold=0,target_sizes=torch.tensor([[480,640]]))[0]['boxes'].numpy()
    np.testing.assert_allclose(engine.boxes[0],official_boxes,atol=1e-3,rtol=1e-5)

from scenequery.common import ROOT,DATA,read_json,sha256
import numpy as np
from PIL import Image


def test_split_and_input_integrity():
    frames=read_json(ROOT/'manifests/frames.json')['frames']
    training={f['rgb'] for f in frames if f['split']=='train'}
    heldout={f['rgb'] for f in frames if f['split']=='heldout'}
    assert not training&heldout
    assert len(training)==48 and len(heldout)==12
    for f in frames:
        assert abs(f['timestamp']-f['depth_timestamp'])<=.02
        assert sha256(DATA/f['rgb'])==f['rgb_sha256']
        assert sha256(DATA/f['depth'])==f['depth_sha256']
        assert Image.open(DATA/f['rgb']).size==(640,480)
        assert Image.open(DATA/f['depth']).size==(640,480)


def test_eval_annotations_only_reference_heldout_frames():
    frames=read_json(ROOT/'manifests/frames.json')['frames']
    annotations=read_json(ROOT/'config/evaluation_annotations.json')['annotations']
    for a in annotations:assert frames[a['frame_id']]['split']=='heldout'

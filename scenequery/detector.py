"""Pinned frozen detector selection and its documented query preprocessing."""
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
from .common import ROOT,read_json


def settings():return read_json(ROOT/'config/model.json')


def load_detector():
    config=settings();path=ROOT/config['directory']
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(path,local_files_only=True).eval()
    for p in model.parameters():p.requires_grad_(False)
    return processor,model

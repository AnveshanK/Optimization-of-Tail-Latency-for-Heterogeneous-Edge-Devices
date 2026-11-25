"""
The three edge services, used by the worker processes. Each request runs in its own process
with its own copy of the model (loaded once, before the timed processing starts), and process()
handles one input file, returning per-stage timings in milliseconds.

  d  object detection       YOLOv5s          (same code path as yolov5/detect.py)
  p  instance segmentation  YOLOv5s-seg      (same code path as yolov5/segment/predict.py)
  s  speech-to-text         Wav2Vec2-large-960h (same model as the earlier speech.py)

This module imports torch, so it is only imported inside worker processes.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolov5"
MODELS = ROOT / "models"

IMG_SIZE = 640
CONF_THRES, IOU_THRES, MAX_DET = 0.25, 0.45, 1000   # yolov5 defaults


def _sync(device):
    # finish all queued GPU work before taking a timestamp
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ensure_yolo_path():
    if str(YOLO_DIR) not in sys.path:
        sys.path.insert(0, str(YOLO_DIR))


def prepare_imports(task):
    """Import the model code of a task before the measurement (as on a server whose process is
    already running), so the timed model load only covers reading the weights and building it."""
    if task in ("d", "p"):
        _ensure_yolo_path()
        import cv2  # noqa: F401
        import models.common  # noqa: F401
        import utils.augmentations  # noqa: F401
        import utils.general  # noqa: F401
        import utils.segment.general  # noqa: F401
    else:
        import soundfile  # noqa: F401
        import transformers  # noqa: F401
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: F401


def _ms(a, b):
    return (b - a) * 1000.0


class _YoloBase:
    weights = None

    def __init__(self, device):
        _ensure_yolo_path()
        import cv2
        from models.common import DetectMultiBackend
        from utils.augmentations import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_boxes

        self.cv2, self.letterbox = cv2, letterbox
        self.nms, self.scale_boxes = non_max_suppression, scale_boxes
        self.device = device
        self.model = DetectMultiBackend(str(MODELS / self.weights), device=device, fp16=False)
        self.stride, self.pt = self.model.stride, self.model.pt
        self.imgsz = check_img_size(IMG_SIZE, s=self.stride)
        self.model.eval()

    def _prepare(self, path):
        t0 = time.perf_counter()
        im0 = self.cv2.imread(str(path))  # BGR
        t1 = time.perf_counter()
        im = self.letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC->CHW, BGR->RGB
        im = torch.from_numpy(im).to(self.device).float() / 255.0
        im = im[None]
        _sync(self.device)
        t2 = time.perf_counter()
        return im0, im, t0, t1, t2


class DetectTask(_YoloBase):
    weights = "yolov5s.pt"

    @torch.inference_mode()
    def process(self, path):
        im0, im, t0, t1, t2 = self._prepare(path)
        pred = self.model(im)
        _sync(self.device)
        t3 = time.perf_counter()
        det = self.nms(pred, CONF_THRES, IOU_THRES, max_det=MAX_DET)[0]
        if len(det):
            det[:, :4] = self.scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        out = det.cpu().numpy()  # result returned to the user: boxes, conf, class
        t4 = time.perf_counter()
        return {"read_ms": _ms(t0, t1), "preprocess_ms": _ms(t1, t2), "inference_ms": _ms(t2, t3),
                "postprocess_ms": _ms(t3, t4), "n_objects": int(len(out)), "output_bytes": int(out.nbytes)}


class SegmentTask(_YoloBase):
    weights = "yolov5s-seg.pt"

    def __init__(self, device):
        super().__init__(device)
        from utils.segment.general import process_mask
        self.process_mask = process_mask

    @torch.inference_mode()
    def process(self, path):
        im0, im, t0, t1, t2 = self._prepare(path)
        pred, proto = self.model(im)[:2]
        _sync(self.device)
        t3 = time.perf_counter()
        det = self.nms(pred, CONF_THRES, IOU_THRES, max_det=MAX_DET, nm=32)[0]
        out_bytes = 0
        if len(det):
            masks = self.process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
            det[:, :4] = self.scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            mask_bits = np.packbits((masks > 0.5).cpu().numpy())  # masks returned to the user
            out_bytes = int(det[:, :6].cpu().numpy().nbytes + mask_bits.nbytes)
        else:
            _sync(self.device)
        t4 = time.perf_counter()
        return {"read_ms": _ms(t0, t1), "preprocess_ms": _ms(t1, t2), "inference_ms": _ms(t2, t3),
                "postprocess_ms": _ms(t3, t4), "n_objects": int(len(det)), "output_bytes": out_bytes}


class SpeechTask:
    SR = 16000

    def __init__(self, device):
        import soundfile as sf
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from transformers.utils import logging as hf_logging

        # 'masked_spec_embed not initialized' is a training-only weight; the warning is harmless
        hf_logging.set_verbosity_error()

        model_dir = MODELS / "wav2vec2-large-960h"
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} missing - run run.py so it downloads the model")
        self.sf, self.device = sf, device
        self.processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
        self.model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device).eval()

    def _load(self, path):
        audio, sr = self.sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.SR:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, self.SR, sr).astype(np.float32)
        return audio

    @torch.inference_mode()
    def process(self, path):
        t0 = time.perf_counter()
        audio = self._load(path)
        t1 = time.perf_counter()
        x = self.processor(audio, sampling_rate=self.SR, return_tensors="pt", padding=True).input_values.to(self.device)
        _sync(self.device)
        t2 = time.perf_counter()
        logits = self.model(x).logits
        _sync(self.device)
        t3 = time.perf_counter()
        ids = torch.argmax(logits, dim=-1)
        text = self.processor.decode(ids[0].cpu())
        t4 = time.perf_counter()
        return {"read_ms": _ms(t0, t1), "preprocess_ms": _ms(t1, t2), "inference_ms": _ms(t2, t3),
                "postprocess_ms": _ms(t3, t4), "n_objects": len(text.split()),
                "output_bytes": len(text.encode("utf-8"))}


TASK_CLASSES = {"d": DetectTask, "p": SegmentTask, "s": SpeechTask}


def feature_reader(task):
    """Function path -> input features of one file (size, resolution or duration).
    Used before t0, so it does not need the model."""
    import os

    if task in ("d", "p"):
        import cv2

        def read(path):
            h, w = cv2.imread(str(path)).shape[:2]
            return {"input_bytes": os.path.getsize(path), "img_width": w, "img_height": h, "img_pixels": w * h}
    else:
        import soundfile as sf

        def read(path):
            info = sf.info(str(path))
            return {"input_bytes": os.path.getsize(path), "audio_duration_s": float(info.duration),
                    "audio_samples": int(info.frames)}
    return read

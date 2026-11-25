# Optimization of Tail Latency for Heterogeneous Edge Devices

B.Tech Project (Research track), IIIT-Delhi. Students: Anveshan Khanna, Ramneek Uttam. Advisor: Dr. Arani Bhattacharya.

The project studies **batch redundancy scheduling for tail latency on heterogeneous edge servers**: at every decision, an
edge controller chooses which waiting requests to send now and to which group of servers each one goes (redundant
copies, first result wins), so that p95/p99 latency and missed deadlines stay low without wasting server time. The
proposed scheduler is a Transformer-based model that builds each decision edge by edge, trained with meta-reinforcement
learning (PPO) over many randomly generated edge systems and deployed with frozen weights.

## Contents

| Folder | What it is |
|---|---|
| [`Convergence_Check/`](Convergence_Check/) | Implementation of the proposed framework and an event-driven simulator, used to check on synthetic data that training converges and learns the right behaviour (Chapter 8 of the report). Includes the report of the check. |
| [`Dataset_Collection_Tool/`](Dataset_Collection_Tool/) | Tool that measures the per-input latency of three ML services (speech recognition, object detection, segmentation) on a machine with 1–5 requests running at the same time (Chapter 9 of the report). |

Each folder has its own README with setup and run instructions.

## Third-party content

- `Dataset_Collection_Tool/yolov5/` and the YOLOv5 weights are from [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) (AGPL-3.0, see its `LICENSE`).
- The input images are from [COCO](https://cocodataset.org) and the audio clips from [LibriSpeech](https://www.openslr.org/12) test-clean (both CC BY 4.0).

# Smart Parking: Edge Computing Node

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Object_Detection-00FFFF?logo=yolo&logoColor=black)
![Raspberry Pi](https://img.shields.io/badge/Edge-Raspberry_Pi-C51A4A?logo=raspberry-pi&logoColor=white)

This repository contains the source code for the local processing node (Edge Node) of a smart parking system, developed as part of a Master's thesis.

The project focuses on extracting real-time parking occupancy data using Computer Vision techniques in a resource-constrained environment (Raspberry Pi). The node processes video streams locally and syncs the required metadata with a cloud-based PostgreSQL database.

## System Architecture and Modules

The codebase is structured to ensure operational resilience and low resource consumption. The core modules include:

### 1. Real-Time Monitoring (`main2.py`)
The primary inference script running continuously on the Edge device.
* **Tracking and Detection:** Uses YOLO models integrated with the BoT-SORT algorithm.
* **Spatio-Temporal Logic:** Implements a state machine with hysteresis (different frame thresholds for entry and exit) and a kinematic filter (apparent zero velocity requirement). This rejects temporary occlusions, brief maneuvers, and passing traffic.
* **Database Synchronization:** Handles transactional updates of parking spot states and historical logs in PostgreSQL.

### 2. HITL Calibration and Mapping (`generator_linebased.py`)
Spatial calibration tool to define parking geometries.
* **Line-Based Heatmaps:** Generates automatic parking spot proposals based on the statistical accumulation of vehicle contact points (tires), minimizing false positives caused by perspective distortion.
* **Human-in-the-Loop (HITL):** Bidirectional adjustment interface and manual drawing tool. Allows the administrator to iteratively correct and fine-tune the AI's spatial inferences.

### 3. Data Collection and Capture (`logger.py` and `get_frames_yt.py`)
* **Universal Logger:** Resolution-agnostic system for recording occupancy history. Includes tracking ID corruption prevention through session offsets.
* **Stream Capture:** Isolated script for video acquisition designed to handle network drops and corrupted video streams without crashing the main inference pipeline.

## Technical Highlights
* **Edge Operation:** All heavy visual processing (inference, tracking, polygon calculation) is performed locally. Only metadata is transmitted to the cloud, significantly reducing bandwidth requirements.
* **Fault Tolerance:** Built-in safeguards for race conditions during image reading (`\xff\xd9` EOF checks) and temporary database connection failures.

---
**Author:** Rúben Bandeira

*Note: The Next.js web application developed for the end-user interface and system management is stored in a separate private repository due to environment variables and security constraints.*

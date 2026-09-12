# ?? Garud-Drishti (UKSI P-007)

**Garud-Drishti** is an **AI-powered crowd safety analytics system** ? designed to monitor large gatherings from drone or CCTV camera feeds. It acts as a real-time control room that evaluates crowd dynamics and warns operators **before** a dangerous crush or stampede develops. ???

Originally modeled for dense pilgrimage scenarios (ghats, temples ??), it is equally highly effective for concerts ??, stadiums ???, transport hubs ??, and protests ??????????.

---

## ?? Key Features

* **?? Smart Density Estimation:** Uses Deep Learning (CSRNet), YOLO object detection, or classical blob tracking to calculate exact crowd density (persons/m²) across different zones. ??
* **?? Optical Flow Analysis:** Tracks dense crowd movement, speed, direction, and flow coherence in real time. ??
* **?? 5-Signal Risk Engine:** Evaluates independent danger signals:
  * ?? **Density:** Basic overcrowding.
  * ? **Density Trend:** Rate of density change & time-to-critical forecasting.
  * ?? **Flow Breakdown:** Detects when a crowd is packing in but stalling (the #1 precursor to a crush).
  * ?? **Pressure:** Measures crowd turbulence (density × velocity variance).
  * ?? **Convergence:** Detects opposing counter-flows causing dangerous compression.
* **?? Smart Alert Manager:** Hysteresis and dwell timers prevent alarm fatigue and false positives. Auto-escalates severe risks. ??
* **??? Live Dashboard:** Web-based control room UI showing real-time heat maps ??, flow vectors ??, active zones, and alerts with evidence snapshots ??.
* **?? Privacy by Design:** Analyzes crowds as fluid dynamics. No facial recognition, no individual trajectory tracking, and automatic data purging. ???

---

## ??? Setup & Installation

### ?? Prerequisites
* Python 3.12+ ??
* Windows ?? / Linux ?? / macOS ??

### 1?? Create and Activate Virtual Environment
`ash
python -m venv .venv

# On Windows:
.venv\Scripts\Activate.ps1
# On Mac/Linux:
source .venv/bin/activate
`

### 2?? Install Dependencies ??
`ash
pip install fastapi uvicorn pydantic numpy opencv-python scipy
`
*(Optional: Install 	orch, 	orchvision, and ultralytics for YOLO/CSRNet deep learning backends).*

### 3?? Run the Application ?????
`ash
python run.py
`
* **Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/) ???
* **Setup UI:** [http://127.0.0.1:8000/setup.html](http://127.0.0.1:8000/setup.html) ??

---

## ?? Using Your Own Video Data

To run the pipeline on your own drone ?? or CCTV ?? footage, simply point it to a local video file or an RTSP live stream:

`ash
# Use a local video file ??
python run.py --source "C:\path\to\your\video.mp4"

# Use a live IP Camera / Drone stream ??
python run.py --source "rtsp://192.168.1.100:554/stream"
`

### ?? Scene Calibration
After launching with your custom video, open the **Setup UI** (/setup.html) to calibrate the system for your camera's perspective:
1. **?? Ground Calibration:** Click 4 points on the ground to form a rectangle, and enter real-world dimensions (metres). This translates pixels to real ground area.
2. **??? Draw Zones:** Draw polygons over areas you want to monitor.
3. **?? Draw Corridors:** Mark specific evacuation routes or narrow choke points.

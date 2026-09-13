# Garud-Drishti (UKSI P-007)
**AI-Powered Crowd Safety & Risk Analytics Platform**

Garud-Drishti is a real-time computer vision system designed to monitor large gatherings from drone or CCTV camera feeds. It acts as an intelligent control room that evaluates crowd dynamics, density, and flow, warning operators **before** a dangerous crush or stampede can develop.

Originally modeled for dense pilgrimage scenarios (ghats, temples), it is highly adaptable for concerts, stadiums, transport hubs, and public protests.

---

## Core Capabilities

* **Smart Density Estimation:** Uses Deep Learning (CSRNet), YOLO object detection, or classical blob tracking to calculate precise crowd density (persons/m²) across custom zones.
* **Optical Flow Analysis:** Tracks dense crowd movement, speed, direction, and flow coherence in real-time, stripping out drone/camera jitter.
* **5-Signal Risk Engine:** Evaluates independent danger signals simultaneously:
  1. **Density:** Basic overcrowding thresholds.
  2. **Density Trend:** Rate of density change & time-to-critical forecasting.
  3. **Flow Breakdown:** Detects when a crowd is packing in but stalling (the #1 precursor to a crowd crush).
  4. **Pressure:** Measures crowd turbulence (density × velocity variance).
  5. **Convergence:** Detects opposing counter-flows causing dangerous compression.
* **Smart Alert Manager:** Uses hysteresis and dwell timers to prevent alarm fatigue and false positives. Auto-escalates severe risks.
* **Live Dashboard:** Web-based control room UI showing real-time heat maps, flow vectors, active zones, and alerts with evidence snapshots.
* **Privacy by Design:** Analyzes crowds as fluid dynamics. No facial recognition, no individual trajectory tracking, and automatic evidence purging.

---

## System Requirements

### Hardware
* **Minimum (Testing/Demo):** Standard modern CPU (Intel i5/Ryzen 5 or better), 8GB RAM. 
* **Recommended (Production):** Dedicated NVIDIA GPU (for running YOLO/CSRNet on live, high-resolution feeds), 16GB+ RAM.

### Software
* **OS:** Windows, macOS, or Linux
* **Environment:** Python 3.12 or newer
* **Dependencies:** FastAPI, Uvicorn, Pydantic, NumPy, OpenCV, SciPy

---

## Getting Started

### 1. Setup Environment
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/Mayankparmar007/Garud-Drishti.git
cd Garud-Drishti

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1
# Activate (Mac/Linux)
source .venv/bin/activate
```

### 2. Install Packages
Install the core requirements for the pipeline and dashboard:
```bash
pip install fastapi uvicorn pydantic numpy opencv-python scipy
```
*(Optional) If you plan to process real footage using advanced AI, install the deep learning backends:*
```bash
pip install torch torchvision ultralytics
```

---

## Usage

Garud-Drishti comes with a built-in "Synthetic Demo" so you can test the dashboard immediately, or you can plug in your own video feeds.

### Run the Application
```bash
# Option A: Run the built-in synthetic demo (No video required)
python run.py

# Option B: Run on a local video file
python run.py --source "C:\path\to\your\video.mp4"

# Option C: Run on a live Drone/CCTV IP stream
python run.py --source "rtsp://192.168.1.100:554/stream"
```

### Access the Interfaces
Once running, open your web browser:
* **Live Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **Scene Setup & Calibration:** [http://127.0.0.1:8000/setup.html](http://127.0.0.1:8000/setup.html)

---

## Calibration & Configuration

To use Garud-Drishti on your own footage, you must calibrate the system to understand your camera's perspective. 

1. Launch the app with your video source.
2. Open the **Setup UI** (`/setup.html`).
3. **Ground Calibration:** Click 4 points on the ground to form a rectangle, and enter the real-world dimensions (in metres). This translates flat pixels to real-world ground area.
4. **Draw Zones:** Draw polygons over specific areas you want to monitor (e.g., "Main Gate", "Platform 1").
5. **Draw Corridors:** Mark specific evacuation routes or narrow choke points.
6. Click **Save** to instantly apply the new configuration to the live dashboard.

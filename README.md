# Smart rPPG - Remote Vital Signs Monitor

Non-contact heart rate, respiratory rate, and stress monitoring using Raspberry Pi + Camera Module 3.

## What it does
- Measures HR and RR using face and chest
- Uses ICA + CHROM/POS for cleaner PPG signal
- Real-time peak detection + PPG waveform graph
- Detects motion and lighting issues
- Estimates stress level from HRV (RMSSD)

## Hardware
- Raspberry Pi 4B (recommended)
- Raspberry Pi Camera Module 3
- 5V 3A power supply

## Installation
1. Update Pi:
   sudo apt update && sudo apt upgrade -y
   sudo apt install python3-opencv python3-picamera2 -y

2. Install Python packages:
   pip install numpy scipy opencv-python picamera2

3. Download and run:
   git clone https://github.com/phonot/rPPG_signs_monitoring.git
   cd smart-rppg-vitals
   python3 hr_rr_hrv.py.py

## How to use
- Runs for 60 seconds by default
- Press Q to exit
- Stay still, face the camera
- Live display shows HR, RR, PPG graph, and quality

## Sample Output
[✓] 58s | HR:  72 BPM, RR: 16/min | Stress: MODERATE STRESS | Stable

Final Results:
Heart Rate: 73 BPM
Respiratory Rate: 15 breaths/min
RMSSD: 67.8 ms
Stress: MODERATE STRESS

##picture
![Project Prototype](https://raw.githubusercontent.com/phonot/rPPG_signs_monitoring/main/my_pi4.png)

## License
MIT License

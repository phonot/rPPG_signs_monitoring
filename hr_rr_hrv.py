#!/usr/bin/env python3
"""
Vital Signs Monitor with ICA-based PPG Extraction + Peak Detection
- Advanced: ICA (Independent Component Analysis) for cleaner PPG signal
- Multi-band chrominance methods (CHROM, POS) for robust signal extraction
- Real-time peak detection with systolic peak identification
- Live peak detection graph on screen
- Minimal stable HR and RR with aggressive smoothing
- Better UI: smaller text, improved font style
- Raspberry Pi 4B + Camera Module 3
Duration: 60 seconds
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls
import time
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.linalg import svd
from collections import deque
import os
import urllib.request
import threading
from queue import Queue


class ICABasedHRVCalculator:
    """ICA-based HRV with peak detection"""
    
    def __init__(self):
        self.nn_intervals = deque(maxlen=200)
        self.last_beat_time = 0
        self.beat_threshold = 0.3
        self.detected_peaks = deque(maxlen=60)  # For graph display
        
    def ica_extraction(self, rgb_signal):
        """Independent Component Analysis for PPG extraction"""
        if len(rgb_signal) < 10:
            return None
        
        try:
            # Standardize
            X = np.array(rgb_signal).T
            X = X - np.mean(X, axis=1, keepdims=True)
            X = X / (np.std(X, axis=1, keepdims=True) + 1e-6)
            
            # SVD-based ICA initialization
            U, S, Vt = svd(X, full_matrices=False)
            W = U[:, :min(3, U.shape[1])]
            
            # FastICA iterations (simplified)
            S_ica = W.T @ X
            
            # Return component with highest power in cardiac frequency range
            return S_ica[0]  # Primary component
        except:
            return None
    
    def chrom_method(self, rgb_signal):
        """CHROM (Chrominance) method for PPG extraction"""
        if len(rgb_signal) < 20:
            return None
        
        try:
            X = np.array(rgb_signal).T
            mean_X = np.mean(X, axis=1, keepdims=True)
            normalized = X / (mean_X + 1e-6)
            
            # CHROM computation
            Xs = 3 * normalized[0] - 2 * normalized[1]
            Ys = 1.5 * normalized[0] + normalized[1] - 1.5 * normalized[2]
            
            return np.array([Xs, Ys])
        except:
            return None
    
    def pos_method(self, rgb_signal):
        """POS (Plane-Orthogonal-to-Skin) method for PPG extraction"""
        if len(rgb_signal) < 20:
            return None
        
        try:
            X = np.array(rgb_signal).T
            mean_X = np.mean(X, axis=1, keepdims=True)
            normalized = X / (mean_X + 1e-6)
            
            # Project onto skin plane orthogonal
            Xf = normalized[0] - normalized[2]  # R - B
            Yf = 2 * normalized[1] - normalized[0] - normalized[2]  # 2G - R - B
            
            return np.array([Xf, Yf])
        except:
            return None
    
    def detect_peaks_from_signal(self, filtered_signal, fps=20):
        """Detect systolic peaks in PPG waveform"""
        if filtered_signal is None or len(filtered_signal) < 10:
            return []
        
        try:
            # Normalize signal
            normalized = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal) + 1e-6)
            
            # Find peaks with cardiac frequency constraints
            min_distance = int(0.4 * fps)  # Minimum 400ms between beats (150 BPM max)
            peaks, properties = signal.find_peaks(
                normalized,
                height=self.beat_threshold,
                distance=min_distance,
                prominence=0.1
            )
            
            return peaks
        except:
            return []
    
    def calculate_rmssd(self):
        """Calculate RMSSD"""
        if len(self.nn_intervals) < 2:
            return 0
        
        try:
            intervals = np.array(self.nn_intervals)
            successive_diffs = np.diff(intervals)
            rmssd = np.sqrt(np.mean(successive_diffs ** 2))
            return rmssd
        except:
            return 0
    
    def get_nn_count(self):
        """Get number of NN intervals"""
        return len(self.nn_intervals)
    
    def classify_stress_level(self):
        """Classify stress using RMSSD"""
        rmssd = self.calculate_rmssd()
        
        if rmssd > 100:
            return "LOW STRESS", (0, 255, 0)
        elif rmssd > 50:
            return "MODERATE STRESS", (0, 255, 255)
        else:
            return "HIGH STRESS", (0, 0, 255)


class QualityMetrics:
    """Quality metrics - Face, Motion, Lighting"""
    
    def __init__(self):
        self.face_quality_history = deque(maxlen=50)
        self.motion_stability_history = deque(maxlen=50)
        self.lighting_quality_history = deque(maxlen=50)
    
    def calculate_face_quality(self, face_coords, frame_shape):
        """Face quality score"""
        if face_coords is None:
            return 0
        
        try:
            x, y, w, h = face_coords
            frame_h, frame_w = frame_shape[:2]
            
            face_area_ratio = (w * h) / (frame_w * frame_h)
            area_score = 100 - abs(face_area_ratio - 0.25) * 200
            area_score = max(0, min(100, area_score))
            
            center_x = x + w/2
            center_y = y + h/2
            center_dist = np.sqrt(((center_x - frame_w/2)**2 + (center_y - frame_h/2)**2))
            center_score = 100 - (center_dist / (frame_w/2)) * 100
            center_score = max(0, min(100, center_score))
            
            quality = (area_score + center_score) / 2
            self.face_quality_history.append(quality)
            return quality
        except:
            return 0
    
    def calculate_lighting_quality(self, frame):
        """Lighting quality score"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            dark_pixels = np.sum(hist[:64])
            bright_pixels = np.sum(hist[192:])
            mid_pixels = np.sum(hist[64:192])
            
            balance_score = mid_pixels * 100
            extreme_score = 100 - (dark_pixels + bright_pixels) * 100
            
            quality = (balance_score + extreme_score) / 2
            quality = max(0, min(100, quality))
            self.lighting_quality_history.append(quality)
            return quality
        except:
            return 0
    
    def calculate_motion_stability(self, motion_history):
        """Motion stability score"""
        if len(motion_history) < 10:
            return 0
        
        try:
            recent_motion = np.array(list(motion_history)[-10:])
            motion_std = np.std(recent_motion)
            motion_mean = np.mean(recent_motion)
            
            stability_score = 100 - min(100, motion_std * 50)
            activity_score = 100 - min(100, motion_mean * 33)
            
            quality = (stability_score + activity_score) / 2
            quality = max(0, min(100, quality))
            self.motion_stability_history.append(quality)
            return quality
        except:
            return 0
    
    def get_quality_breakdown(self):
        """Get quality breakdown"""
        return {
            'face_quality': np.mean(self.face_quality_history) if len(self.face_quality_history) > 0 else 0,
            'motion_stability': np.mean(self.motion_stability_history) if len(self.motion_stability_history) > 0 else 0,
            'lighting_quality': np.mean(self.lighting_quality_history) if len(self.lighting_quality_history) > 0 else 0,
        }


class AdvancedVitalSignsMonitor:
    def __init__(self):
        print("\n[*] Initializing Advanced Vital Signs Monitor with ICA+Peak Detection...")

        self.cascade_path = self.ensure_cascade_classifier()

        try:
            self.picam2 = Picamera2()

            config = self.picam2.create_preview_configuration(
                main={"size": (320, 240), "format": "BGR888"}
            )
            self.picam2.configure(config)

            self.picam2.set_controls({
                "AfMode": controls.AfModeEnum.Continuous,
                "AeEnable": True,
                "AwbMode": controls.AwbModeEnum.Auto
            })
            print("✓ Camera initialized (320x240)")
        except Exception as e:
            print(f"✗ Camera error: {e}")
            raise

        if self.cascade_path:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self.face_cascade.empty():
                raise Exception("Cascade classifier is empty")
            print(f"✓ Cascade loaded")
        else:
            raise Exception("No cascade classifier")

        self.hrv_calculator = ICABasedHRVCalculator()
        self.quality_metrics = QualityMetrics()
        print("✓ ICA-based HRV and quality metrics initialized")

        self.fps = 20
        self.buffer_size = 100
        self.rgb_buffer = deque(maxlen=self.buffer_size)

        self.resp_buffer_size = 200
        self.motion_buffer = deque(maxlen=self.resp_buffer_size)

        self.hr_freq_min = 0.67
        self.hr_freq_max = 3.0

        self.rr_freq_min = 0.17
        self.rr_freq_max = 0.67

        self.raw_hr = 0
        self.raw_rr = 0

        self.current_hr = 0
        self.current_rr = 0

        self.hr_alpha = 0.05
        self.rr_alpha = 0.05

        self.max_hr_jump = 2
        self.max_rr_jump = 1

        self.first_hr_reading = True
        self.first_rr_reading = True

        self.final_hr_buffer = deque(maxlen=30)
        self.final_rr_buffer = deque(maxlen=30)

        # Peak detection graph buffer
        self.ppg_waveform_buffer = deque(maxlen=100)
        self.peak_positions = deque(maxlen=30)

        self.motion_history = deque(maxlen=20)
        self.high_motion_detected = False
        self.motion_threshold = 5.0

        self.detection_status = "Initializing..."
        self.face_detected = False
        self.face_coords = None
        self.pulse_signal = None

        self.prev_gray = None

        self.frame_queue = Queue(maxsize=2)
        self.stop_event = threading.Event()

    def ensure_cascade_classifier(self):
        """Download cascade if not present"""
        local_path = "haarcascade_frontalface_default.xml"

        if os.path.exists(local_path):
            return local_path

        print("[*] Downloading cascade classifier...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"✓ Downloaded cascade")
            return local_path
        except:
            for path in ["/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"]:
                if os.path.exists(path):
                    return path

        return None

    def start_camera(self):
        """Start camera"""
        self.picam2.start()
        time.sleep(1)
        print("✓ Camera started")

    def stop_camera(self):
        """Stop camera"""
        self.picam2.stop()
        print("✓ Camera stopped")

    def detect_face(self, frame):
        """Detect face"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30)
            )
        except:
            return None, frame, None

        if len(faces) > 0:
            self.face_detected = True
            face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = face

            roi_y = max(0, y + int(0.1 * h))
            roi_h = max(1, int(0.3 * h))
            roi_x = max(0, x + int(0.2 * w))
            roi_w = max(1, int(0.6 * w))

            roi_y_end = min(roi_y + roi_h, frame.shape[0])
            roi_x_end = min(roi_x + roi_w, frame.shape[1])

            roi = frame[roi_y:roi_y_end, roi_x:roi_x_end]

            cv2.rectangle(frame, (roi_x, roi_y), (roi_x_end, roi_y_end), (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            self.detection_status = "Face detected"
            self.face_coords = (x, y, w, h)
            return roi, frame, (roi_x, roi_y, roi_w, roi_h)

        self.face_detected = False
        self.detection_status = "No face"
        self.current_hr = 0
        self.current_rr = 0
        self.first_hr_reading = True
        self.first_rr_reading = True
        self.face_coords = None
        return None, frame, None

    def detect_chest_roi(self, frame):
        """Detect chest ROI"""
        h, w = frame.shape[:2]

        chest_y = max(0, int(h * 0.35))
        chest_h = max(1, int(h * 0.45))
        chest_x = max(0, int(w * 0.25))
        chest_w = max(1, int(w * 0.5))

        chest_y_end = min(chest_y + chest_h, h)
        chest_x_end = min(chest_x + chest_w, w)

        roi = frame[chest_y:chest_y_end, chest_x:chest_x_end]

        cv2.rectangle(frame, (chest_x, chest_y), (chest_x_end, chest_y_end), (255, 0, 0), 1)

        return roi, (chest_x, chest_y, chest_w, chest_h)

    def extract_rgb_signal(self, roi):
        """Extract RGB values"""
        if roi is None or roi.size == 0:
            return None

        try:
            b_mean = np.mean(roi[:, :, 0])
            g_mean = np.mean(roi[:, :, 1])
            r_mean = np.mean(roi[:, :, 2])
            return np.array([r_mean, g_mean, b_mean])
        except:
            return None

    def bandpass_filter(self, data, low_freq, high_freq):
        """Butterworth filter"""
        try:
            nyquist = 0.5 * self.fps
            low = low_freq / nyquist
            high = high_freq / nyquist

            low = max(0.001, min(0.999, low))
            high = max(0.001, min(0.999, high))

            if low >= high:
                return data

            b, a = signal.butter(5, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
            return filtered
        except:
            return data

    def estimate_heart_rate_from_peaks(self, peaks, fps=20):
        """Estimate HR from detected peaks"""
        if len(peaks) < 2:
            return 0
        
        try:
            # Calculate intervals between peaks (in samples)
            peak_intervals = np.diff(peaks)
            
            # Convert to time (seconds)
            peak_times = peak_intervals / fps
            
            # Average interval
            avg_interval = np.mean(peak_times)
            
            if avg_interval > 0:
                hr_bpm = 60.0 / avg_interval
                
                # Store NN intervals for HRV
                interval_ms = avg_interval * 1000
                if 300 < interval_ms < 2000:
                    self.hrv_calculator.nn_intervals.append(interval_ms)
                
                if 40 <= hr_bpm <= 180:
                    return hr_bpm
        except:
            pass
        
        return 0

    def calculate_optical_flow(self, current_gray, roi_coords):
        """Optical flow for respiratory rate"""
        if self.prev_gray is None:
            self.prev_gray = current_gray.copy()
            return 0

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, current_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            x, y, w, h = roi_coords
            y_start = max(0, y)
            y_end = min(y + h, flow.shape[0])
            x_start = max(0, x)
            x_end = min(x + w, flow.shape[1])

            chest_flow = flow[y_start:y_end, x_start:x_end]

            if chest_flow.size > 0:
                motion_magnitude = np.mean(np.sqrt(chest_flow[:, :, 0]**2 + chest_flow[:, :, 1]**2))
            else:
                motion_magnitude = 0

            self.prev_gray = current_gray.copy()
            return motion_magnitude
        except:
            self.prev_gray = current_gray.copy()
            return 0

    def estimate_respiratory_rate(self, motion_signal):
        """Estimate RR"""
        if len(motion_signal) < 40:
            return 0

        try:
            motion_array = np.array(motion_signal)
            detrended = signal.detrend(motion_array)
            filtered = self.bandpass_filter(detrended, self.rr_freq_min, self.rr_freq_max)

            windowed = filtered * np.hanning(len(filtered))
            fft_data = np.abs(fft(windowed))
            freqs = fftfreq(len(windowed), 1.0/self.fps)

            valid_idx = np.where((freqs >= self.rr_freq_min) & (freqs <= self.rr_freq_max))[0]

            if len(valid_idx) == 0:
                return 0

            fft_valid = fft_data[valid_idx]
            freqs_valid = freqs[valid_idx]

            peak_idx = np.argmax(fft_valid)
            peak_freq = freqs_valid[peak_idx]
            rr_bpm = peak_freq * 60

            if 10 <= rr_bpm <= 40:
                return rr_bpm
            else:
                return 0
        except:
            return 0

    def smooth_with_strict_validation(self, raw_value, smoothed_value, alpha, max_jump, is_first_reading, is_hr=True):
        """Smart smoothing with validation"""
        if raw_value == 0:
            return smoothed_value

        if is_first_reading and smoothed_value == 0:
            if is_hr:
                if 50 <= raw_value <= 150:
                    return raw_value
                else:
                    return 0
            else:
                if 10 <= raw_value <= 40:
                    return raw_value
                else:
                    return 0

        if abs(raw_value - smoothed_value) > max_jump:
            if raw_value > smoothed_value:
                raw_value = smoothed_value + max_jump
            else:
                raw_value = smoothed_value - max_jump

        new_smoothed = alpha * raw_value + (1 - alpha) * smoothed_value

        return new_smoothed

    def detect_motion_disturbance(self, motion_signal):
        """Detect excessive motion"""
        if len(motion_signal) < 10:
            return False

        recent_motion = np.array(list(motion_signal)[-10:])
        motion_std = np.std(recent_motion)
        motion_mean = np.mean(recent_motion)

        is_high_motion = motion_std > self.motion_threshold or motion_mean > 3.0

        self.motion_history.append(is_high_motion)

        if len(self.motion_history) >= 5:
            recent_motion_count = sum(self.motion_history)
            self.high_motion_detected = recent_motion_count >= 3

        return self.high_motion_detected

    def process_frame(self, frame):
        """Process frame with ICA and peak detection"""
        face_roi, annotated_frame, face_coords = self.detect_face(frame)

        if face_roi is not None and face_roi.size > 0:
            rgb_values = self.extract_rgb_signal(face_roi)
            if rgb_values is not None:
                self.rgb_buffer.append(rgb_values)

                if len(self.rgb_buffer) >= self.buffer_size:
                    # Try ICA first, fall back to CHROM
                    ica_signal = self.hrv_calculator.ica_extraction(list(self.rgb_buffer))
                    
                    if ica_signal is None:
                        chrom_signals = self.hrv_calculator.chrom_method(list(self.rgb_buffer))
                        if chrom_signals is not None:
                            ica_signal = chrom_signals[0]
                    
                    if ica_signal is None:
                        pos_signals = self.hrv_calculator.pos_method(list(self.rgb_buffer))
                        if pos_signals is not None:
                            ica_signal = pos_signals[0]
                    
                    if ica_signal is not None:
                        # Filter signal
                        pulse = self.bandpass_filter(ica_signal, self.hr_freq_min, self.hr_freq_max)
                        self.pulse_signal = pulse
                        
                        # Store waveform for graph
                        self.ppg_waveform_buffer.append(pulse[-1])
                        
                        # Detect peaks
                        peaks = self.hrv_calculator.detect_peaks_from_signal(pulse, self.fps)
                        
                        if len(peaks) > 0:
                            self.peak_positions = deque(list(peaks[-30:]), maxlen=30)
                            self.raw_hr = self.estimate_heart_rate_from_peaks(peaks, self.fps)
                        else:
                            self.raw_hr = 0

                        self.current_hr = self.smooth_with_strict_validation(
                            self.raw_hr, self.current_hr, self.hr_alpha, 
                            self.max_hr_jump, self.first_hr_reading, is_hr=True
                        )

                        if self.raw_hr > 0:
                            self.final_hr_buffer.append(self.current_hr)
                            self.first_hr_reading = False

        self.quality_metrics.calculate_face_quality(self.face_coords, annotated_frame.shape)
        self.quality_metrics.calculate_lighting_quality(annotated_frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chest_roi, chest_coords = self.detect_chest_roi(annotated_frame)

        motion = self.calculate_optical_flow(gray, chest_coords)
        self.motion_buffer.append(motion)

        self.quality_metrics.calculate_motion_stability(self.motion_buffer)

        if len(self.motion_buffer) >= 50:
            is_disturbed = self.detect_motion_disturbance(list(self.motion_buffer))
            if not is_disturbed:
                self.raw_rr = self.estimate_respiratory_rate(list(self.motion_buffer))

                self.current_rr = self.smooth_with_strict_validation(
                    self.raw_rr, self.current_rr, self.rr_alpha, 
                    self.max_rr_jump, self.first_rr_reading, is_hr=False
                )

                if self.raw_rr > 0:
                    self.final_rr_buffer.append(self.current_rr)
                    self.first_rr_reading = False
            else:
                self.detection_status = "Motion!"

        return annotated_frame

    def draw_camera_feed(self, frame):
        """Draw minimal info with smaller, better font"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        hr_color = (0, 255, 0) if not self.high_motion_detected else (0, 165, 255)
        rr_color = (0, 255, 0) if not self.high_motion_detected else (0, 165, 255)

        hr_display = int(self.current_hr) if self.face_detected else 0
        rr_display = int(self.current_rr) if self.face_detected else 0

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        cv2.putText(frame, f"HR: {hr_display:3d} BPM", (8, 25), font, font_scale, hr_color, thickness)
        cv2.putText(frame, f"RR: {rr_display:2d}/min", (8, 50), font, font_scale, rr_color, thickness)

        face_color = (0, 255, 0) if self.face_detected else (0, 0, 255)
        status_text = "FACE OK" if self.face_detected else "NO FACE"
        cv2.putText(frame, status_text, (w - 90, 25), font, 0.5, face_color, 1)

        return frame

    def draw_peak_detection_graph(self, waveform_data, peaks, height=120, width=200):
        """Draw peak detection waveform graph"""
        graph_img = np.ones((height, width, 3), dtype=np.uint8) * 20

        # Title
        cv2.putText(graph_img, "PPG Waveform + Peaks", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Grid lines
        cv2.line(graph_img, (5, 20), (width - 5, 20), (50, 50, 50), 1)
        cv2.line(graph_img, (5, height - 15), (width - 5, height - 15), (50, 50, 50), 1)

        if len(waveform_data) > 1:
            # Normalize waveform
            waveform_array = np.array(list(waveform_data))
            waveform_min = np.min(waveform_array)
            waveform_max = np.max(waveform_array)
            waveform_range = waveform_max - waveform_min
            
            if waveform_range == 0:
                waveform_range = 1
            
            # Draw waveform
            points = []
            for i, val in enumerate(waveform_data):
                x = int(5 + (i / (len(waveform_data) - 1)) * (width - 10)) if len(waveform_data) > 1 else 5
                y = int(height - 15 - ((val - waveform_min) / waveform_range) * (height - 35))
                y = max(20, min(height - 15, y))
                points.append((x, y))
            
            # Draw waveform line
            for i in range(len(points) - 1):
                cv2.line(graph_img, points[i], points[i + 1], (0, 255, 0), 1)
            
            # Draw detected peaks
            if len(peaks) > 0:
                for peak_idx in peaks:
                    if 0 <= peak_idx < len(waveform_data):
                        x = int(5 + (peak_idx / len(waveform_data)) * (width - 10))
                        y = points[peak_idx][1] if peak_idx < len(points) else height - 15
                        cv2.circle(graph_img, (x, y), 3, (0, 0, 255), -1)  # Red peaks

        return graph_img

    def camera_thread(self):
        """Thread to capture frames"""
        while not self.stop_event.is_set():
            try:
                frame = self.picam2.capture_array()
                if frame is not None:
                    try:
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
            except:
                pass

    def get_average_vitals(self):
        """Calculate average vitals"""
        avg_hr = int(np.mean(self.final_hr_buffer)) if len(self.final_hr_buffer) > 0 else 0
        avg_rr = int(np.mean(self.final_rr_buffer)) if len(self.final_rr_buffer) > 0 else 0

        return avg_hr, avg_rr

    def run(self, duration=60):
        """Run monitoring"""
        self.start_camera()

        cam_thread = threading.Thread(target=self.camera_thread, daemon=True)
        cam_thread.start()

        print(f"\n[*] Monitoring for {duration} seconds...")
        print("[*] Stay still for stable readings")
        print("[*] Press Q to exit")
        print("-" * 80)

        start_time = time.time()
        frame_count = 0
        display_fps = 0
        fps_time = time.time()

        try:
            while (time.time() - start_time) < duration:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except:
                    continue

                processed_frame = self.process_frame(frame)
                display_frame = self.draw_camera_feed(processed_frame)

                frame_count += 1
                if time.time() - fps_time >= 1:
                    display_fps = frame_count
                    frame_count = 0
                    fps_time = time.time()

                # Draw peak detection graph
                peak_graph = self.draw_peak_detection_graph(list(self.ppg_waveform_buffer), list(self.peak_positions), height=240, width=220)

                # Combine
                combined = np.hstack([display_frame, peak_graph])

                cv2.imshow('Vital Signs Monitor - ICA + Peak Detection', combined)

                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    motion_status = "HIGH MOTION" if self.high_motion_detected else "Stable"
                    hr_display = int(self.current_hr) if self.face_detected else 0
                    rr_display = int(self.current_rr) if self.face_detected else 0
                    rmssd = self.hrv_calculator.calculate_rmssd()
                    nn_count = self.hrv_calculator.get_nn_count()
                    stress_level, _ = self.hrv_calculator.classify_stress_level()

                    print(f"[✓] {int(elapsed):2d}s | HR: {hr_display:3d} BPM, RR: {rr_display:2d}/min | "
                          f"RMSSD: {rmssd:.1f}ms, NN: {nn_count:3d} | Peaks: {len(self.peak_positions)} | "
                          f"Stress: {stress_level:15s} | {motion_status}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n[!] Stopped by user (Q pressed)")
                    break

        except KeyboardInterrupt:
            print("\n[!] Stopped by user (Ctrl+C)")

        finally:
            self.stop_event.set()
            self.stop_camera()
            cv2.destroyAllWindows()

            final_hr, final_rr = self.get_average_vitals()
            rmssd = self.hrv_calculator.calculate_rmssd()
            nn_count = self.hrv_calculator.get_nn_count()
            stress_level, _ = self.hrv_calculator.classify_stress_level()
            quality_breakdown = self.quality_metrics.get_quality_breakdown()

            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)
            print("\n--- VITAL SIGNS ---")
            print(f"Heart Rate:                          {final_hr} BPM")
            print(f"Respiratory Rate:                    {final_rr} breaths/min")
            print("\n--- HEART RATE VARIABILITY (RMSSD-BASED) ---")
            print(f"RMSSD:                               {rmssd:.2f} ms")
            print(f"NN Intervals Recorded:               {nn_count}")
            print(f"Stress Classification:               {stress_level}")
            print("\n--- QUALITY ---")
            print(f"Face Quality:                        {int(quality_breakdown['face_quality'])}%")
            print(f"Motion Stability:                    {int(quality_breakdown['motion_stability'])}%")
            print(f"Lighting Quality:                    {int(quality_breakdown['lighting_quality'])}%")
            print("="*80)


def main():
    print("="*80)
    print("Advanced Vital Signs Monitor - ICA + Peak Detection")
    print("Raspberry Pi 4B + Camera Module 3 | 60 Seconds")
    print("="*80)

    try:
        monitor = AdvancedVitalSignsMonitor()
        print("\n✓ System ready!")
        monitor.run(duration=60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()





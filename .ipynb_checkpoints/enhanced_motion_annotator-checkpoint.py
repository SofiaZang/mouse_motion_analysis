import sys
import cv2
import json
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QFileDialog, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QGridLayout, QTextEdit, QComboBox, QCheckBox,
    QMessageBox, QProgressBar, QSplitter
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class DraggableTimeline(FigureCanvas):
    """Custom matplotlib canvas with draggable timeline"""
    timeline_moved = pyqtSignal(int)
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 4))
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.current_frame = 0
        self.total_frames = 0
        self.motion_energy = None
        self.onsets = []
        self.onset_types = {}  # 'active' or 'twitch'
        self.onset_validations = {}  # 'accepted', 'rejected', 'edited'
        
        self.timeline_line = None
        self.dragging = False
        self.connect_events()
        
    def connect_events(self):
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left click
            self.dragging = True
            self.current_frame = int(event.xdata)
            self.update_timeline()
            self.timeline_moved.emit(self.current_frame)
            
    def on_release(self, event):
        self.dragging = False
        
    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        if event.xdata is not None:
            self.current_frame = max(0, min(self.total_frames - 1, int(event.xdata)))
            self.update_timeline()
            self.timeline_moved.emit(self.current_frame)
            
    def update_timeline(self):
        if self.timeline_line:
            self.timeline_line.remove()
        self.timeline_line = self.ax.axvline(self.current_frame, color='red', linewidth=2, alpha=0.8)
        self.draw()
        
    def plot_motion_energy(self, motion_energy, onsets=None, onset_types=None):
        self.motion_energy = motion_energy
        self.total_frames = len(motion_energy)
        self.onsets = onsets or []
        self.onset_types = onset_types or {}
        
        self.ax.clear()
        
        # Plot motion energy
        if motion_energy is not None:
            self.ax.plot(motion_energy, color='black', linewidth=1, alpha=0.7)
            
            # Plot onsets with different colors based on type
            for onset in self.onsets:
                onset_type = self.onset_types.get(onset, 'unknown')
                validation = self.onset_validations.get(onset, 'pending')
                
                if onset_type == 'active':
                    color = 'blue'
                    alpha = 0.8
                elif onset_type == 'twitch':
                    color = 'orange'
                    alpha = 0.8
                else:
                    color = 'gray'
                    alpha = 0.5
                    
                # Adjust alpha based on validation status
                if validation == 'rejected':
                    alpha *= 0.3
                elif validation == 'accepted':
                    alpha *= 1.2
                    
                self.ax.axvline(onset, color=color, linestyle='--', alpha=alpha, linewidth=2)
                
        self.ax.set_xlim(0, self.total_frames)
        if motion_energy is not None:
            self.ax.set_ylim(np.min(motion_energy), np.max(motion_energy))
        self.ax.set_ylabel("Motion Energy")
        self.ax.set_xlabel("Frames")
        self.ax.grid(True, alpha=0.3)
        
        # Update timeline
        self.update_timeline()
        
    def set_onset_validation(self, onset, validation):
        self.onset_validations[onset] = validation
        self.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types)

class MotionAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Motion Onset Annotator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        self.current_frame = 0
        self.playback_speed = 1.0
        
        self.motion_energy = None
        self.classified_events = {}
        self.curated_events = {}
        self.onsets = []
        self.onset_types = {}  # 'active' or 'twitch'
        self.current_onset_idx = 0
        
        # Performance tracking
        self.performance_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
        
        self.init_ui()
        self.setup_timer()
        
    def init_ui(self):
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - Video and controls
        left_panel = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel("Load a video to start")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        left_panel.addWidget(self.video_label)
        
        # Video controls
        video_controls = QGroupBox("Video Controls")
        video_layout = QGridLayout()
        
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        
        self.play_btn = QPushButton("Play ▶️")
        self.play_btn.clicked.connect(self.play)
        
        self.pause_btn = QPushButton("Pause ⏸")
        self.pause_btn.clicked.connect(self.pause)
        
        self.stop_btn = QPushButton("Stop ⏹")
        self.stop_btn.clicked.connect(self.stop)
        
        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(0.1, 5.0)
        self.speed_spinbox.setValue(1.0)
        self.speed_spinbox.setSingleStep(0.1)
        self.speed_spinbox.valueChanged.connect(self.update_playback_speed)
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.valueChanged.connect(self.update_fps)
        
        video_layout.addWidget(self.load_video_btn, 0, 0)
        video_layout.addWidget(self.play_btn, 0, 1)
        video_layout.addWidget(self.pause_btn, 0, 2)
        video_layout.addWidget(self.stop_btn, 0, 3)
        video_layout.addWidget(QLabel("Speed:"), 1, 0)
        video_layout.addWidget(self.speed_spinbox, 1, 1)
        video_layout.addWidget(QLabel("FPS:"), 1, 2)
        video_layout.addWidget(self.fps_spinbox, 1, 3)
        
        video_controls.setLayout(video_layout)
        left_panel.addWidget(video_controls)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_moved)
        left_panel.addWidget(self.frame_slider)
        
        # Frame info
        self.frame_info_label = QLabel("Frame: 0 / 0")
        left_panel.addWidget(self.frame_info_label)
        
        # Right panel - Motion energy and annotation
        right_panel = QVBoxLayout()
        
        # Motion energy timeline
        timeline_group = QGroupBox("Motion Energy Timeline")
        timeline_layout = QVBoxLayout()
        
        self.timeline_canvas = DraggableTimeline()
        self.timeline_canvas.timeline_moved.connect(self.timeline_frame_changed)
        timeline_layout.addWidget(self.timeline_canvas)
        
        # Timeline controls
        timeline_controls = QHBoxLayout()
        self.load_me_btn = QPushButton("Load Motion Energy")
        self.load_me_btn.clicked.connect(self.load_motion_energy)
        
        self.load_class_btn = QPushButton("Load Classifications")
        self.load_class_btn.clicked.connect(self.load_classifications)
        
        timeline_controls.addWidget(self.load_me_btn)
        timeline_controls.addWidget(self.load_class_btn)
        timeline_layout.addLayout(timeline_controls)
        
        timeline_group.setLayout(timeline_layout)
        right_panel.addWidget(timeline_group)
        
        # Onset navigation and validation
        onset_group = QGroupBox("Onset Navigation & Validation")
        onset_layout = QVBoxLayout()
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        self.prev_onset_btn = QPushButton("← Prev Onset")
        self.next_onset_btn = QPushButton("Next Onset →")
        self.prev_onset_btn.clicked.connect(self.prev_onset)
        self.next_onset_btn.clicked.connect(self.next_onset)
        nav_layout.addWidget(self.prev_onset_btn)
        nav_layout.addWidget(self.next_onset_btn)
        onset_layout.addLayout(nav_layout)
        
        # Current onset info
        self.onset_info_label = QLabel("No onsets loaded")
        onset_layout.addWidget(self.onset_info_label)
        
        # Validation buttons
        validation_layout = QHBoxLayout()
        self.accept_btn = QPushButton("✓ Accept")
        self.reject_btn = QPushButton("✗ Reject")
        self.edit_btn = QPushButton("✏ Edit/Move")
        
        self.accept_btn.clicked.connect(lambda: self.validate_onset('accepted'))
        self.reject_btn.clicked.connect(lambda: self.validate_onset('rejected'))
        self.edit_btn.clicked.connect(self.enable_edit_mode)
        
        validation_layout.addWidget(self.accept_btn)
        validation_layout.addWidget(self.reject_btn)
        validation_layout.addWidget(self.edit_btn)
        onset_layout.addLayout(validation_layout)
        
        # Edit mode controls
        edit_layout = QHBoxLayout()
        self.shift_left_btn = QPushButton("← Shift Left")
        self.shift_right_btn = QPushButton("Shift Right →")
        self.shift_step_spinbox = QSpinBox()
        self.shift_step_spinbox.setRange(1, 30)
        self.shift_step_spinbox.setValue(1)
        
        self.shift_left_btn.clicked.connect(lambda: self.shift_onset(-self.shift_step_spinbox.value()))
        self.shift_right_btn.clicked.connect(lambda: self.shift_onset(self.shift_step_spinbox.value()))
        
        edit_layout.addWidget(self.shift_left_btn)
        edit_layout.addWidget(QLabel("Step:"))
        edit_layout.addWidget(self.shift_step_spinbox)
        edit_layout.addWidget(self.shift_right_btn)
        onset_layout.addLayout(edit_layout)
        
        onset_group.setLayout(onset_layout)
        right_panel.addWidget(onset_group)
        
        # Save controls
        save_group = QGroupBox("Save & Export")
        save_layout = QHBoxLayout()
        
        self.save_curated_btn = QPushButton("Save Curated Onsets")
        self.save_metrics_btn = QPushButton("Save Performance Metrics")
        self.save_curated_btn.clicked.connect(self.save_curated_onsets)
        self.save_metrics_btn.clicked.connect(self.save_performance_metrics)
        
        save_layout.addWidget(self.save_curated_btn)
        save_layout.addWidget(self.save_metrics_btn)
        save_group.setLayout(save_layout)
        right_panel.addWidget(save_group)
        
        # Performance metrics display
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(100)
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        
        metrics_group.setLayout(metrics_layout)
        right_panel.addWidget(metrics_group)
        
        # Combine panels
        splitter = QSplitter(Qt.Horizontal)
        
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        splitter.addWidget(left_widget)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([600, 800])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        # Initialize edit mode
        self.edit_mode = False
        self.update_edit_mode_ui()
        
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
    def load_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', 'Videos (*.avi *.mp4 *.mov *.mkv *.tiff *.tif)')
        if fname:
            self.video_path = fname
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open video file")
                return
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps_spinbox.setValue(int(self.fps))
            
            self.frame_slider.setMaximum(self.total_frames - 1)
            self.current_frame = 0
            self.show_frame(self.current_frame)
            self.update_frame_info()
            
    def load_motion_energy(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Motion Energy', '', 'CSV/Excel (*.csv *.xlsx *.npy)')
        if fname:
            if fname.endswith('.npy'):
                self.motion_energy = np.load(fname)
            elif fname.endswith('.csv'):
                df = pd.read_csv(fname)
                self.motion_energy = df.select_dtypes(include=[np.number]).iloc[:, 0].values
            else:
                df = pd.read_excel(fname)
                self.motion_energy = df.select_dtypes(include=[np.number]).iloc[:, 0].values
                
            self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types)
            
    def load_classifications(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Classifications', '', 'JSON (*.json)')
        if fname:
            with open(fname, 'r') as f:
                self.classified_events = json.load(f)
                
            # Process classifications
            self.curated_events = {k: [list(pair) for pair in v] for k, v in self.classified_events.items()}
            
            # Extract onsets and their types
            self.onsets = []
            self.onset_types = {}
            
            for event_type, events in self.curated_events.items():
                for onset, _ in events:
                    self.onsets.append(onset)
                    self.onset_types[onset] = event_type
                    
            self.onsets = sorted(self.onsets)
            self.current_onset_idx = 0
            
            # Update timeline
            self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types)
            
            # Go to first onset
            if self.onsets:
                self.goto_onset(0)
                
    def play(self):
        if self.cap is not None:
            interval = int(1000 / (self.fps * self.playback_speed))
            self.timer.start(interval)
            
    def pause(self):
        self.timer.stop()
        
    def stop(self):
        self.timer.stop()
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.show_frame(0)
        
    def update_playback_speed(self, speed):
        self.playback_speed = speed
        if self.timer.isActive():
            interval = int(1000 / (self.fps * self.playback_speed))
            self.timer.start(interval)
            
    def update_fps(self, fps):
        self.fps = fps
        if self.timer.isActive():
            interval = int(1000 / (self.fps * self.playback_speed))
            self.timer.start(interval)
            
    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.show_frame(self.current_frame)
        else:
            self.pause()
            
    def show_frame(self, frame_num):
        if self.cap is None:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        self.update_frame_info()
        
    def slider_moved(self, value):
        self.current_frame = value
        self.show_frame(value)
        self.timeline_canvas.current_frame = value
        self.timeline_canvas.update_timeline()
        
    def timeline_frame_changed(self, frame):
        self.current_frame = frame
        self.frame_slider.setValue(frame)
        self.show_frame(frame)
        
    def update_frame_info(self):
        self.frame_info_label.setText(f"Frame: {self.current_frame} / {self.total_frames}")
        
    def goto_onset(self, idx):
        if 0 <= idx < len(self.onsets):
            self.current_onset_idx = idx
            onset_frame = self.onsets[idx]
            self.current_frame = onset_frame
            self.frame_slider.setValue(onset_frame)
            self.show_frame(onset_frame)
            self.timeline_canvas.current_frame = onset_frame
            self.timeline_canvas.update_timeline()
            self.update_onset_info()
            
    def prev_onset(self):
        if self.current_onset_idx > 0:
            self.goto_onset(self.current_onset_idx - 1)
            
    def next_onset(self):
        if self.current_onset_idx < len(self.onsets) - 1:
            self.goto_onset(self.current_onset_idx + 1)
            
    def update_onset_info(self):
        if not self.onsets:
            self.onset_info_label.setText("No onsets loaded")
            return
            
        current_onset = self.onsets[self.current_onset_idx]
        onset_type = self.onset_types.get(current_onset, 'unknown')
        validation = self.timeline_canvas.onset_validations.get(current_onset, 'pending')
        
        info_text = f"Onset {self.current_onset_idx + 1}/{len(self.onsets)}: Frame {current_onset}"
        info_text += f" | Type: {onset_type} | Status: {validation}"
        
        self.onset_info_label.setText(info_text)
        
    def validate_onset(self, validation):
        if not self.onsets:
            return
            
        current_onset = self.onsets[self.current_onset_idx]
        self.timeline_canvas.set_onset_validation(current_onset, validation)
        
        # Update performance metrics
        if validation == 'accepted':
            self.performance_metrics['true_positives'] += 1
        elif validation == 'rejected':
            self.performance_metrics['false_positives'] += 1
            
        self.update_performance_display()
        self.update_onset_info()
        
        # Auto-advance to next onset
        if self.current_onset_idx < len(self.onsets) - 1:
            self.next_onset()
            
    def enable_edit_mode(self):
        self.edit_mode = not self.edit_mode
        self.update_edit_mode_ui()
        
    def update_edit_mode_ui(self):
        self.shift_left_btn.setEnabled(self.edit_mode)
        self.shift_right_btn.setEnabled(self.edit_mode)
        
        if self.edit_mode:
            self.edit_btn.setStyleSheet("background-color: lightblue")
        else:
            self.edit_btn.setStyleSheet("")
            
    def shift_onset(self, shift):
        if not self.edit_mode or not self.onsets:
            return
            
        current_onset = self.onsets[self.current_onset_idx]
        new_frame = max(0, min(self.total_frames - 1, current_onset + shift))
        
        # Update onset in data structures
        old_onset = current_onset
        self.onsets[self.current_onset_idx] = new_frame
        
        # Update in curated events
        for event_type, events in self.curated_events.items():
            for event in events:
                if event[0] == old_onset:
                    event[0] = new_frame
                    self.onset_types[new_frame] = self.onset_types.pop(old_onset, event_type)
                    break
                    
        # Update timeline
        self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types)
        
        # Go to the updated onset
        self.goto_onset(self.onsets.index(new_frame))
        
    def update_performance_display(self):
        metrics = self.performance_metrics
        total = sum(metrics.values())
        
        if total > 0:
            precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
            recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0
            
        display_text = f"""
Performance Metrics:
- True Positives: {metrics['true_positives']}
- False Positives: {metrics['false_positives']}
- True Negatives: {metrics['true_negatives']}
- False Negatives: {metrics['false_negatives']}
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1-Score: {f1:.3f}
        """
        
        self.metrics_text.setText(display_text.strip())
        
    def save_curated_onsets(self):
        if not self.curated_events:
            QMessageBox.warning(self, "Warning", "No onsets to save")
            return
            
        # Create corrected onsets based on validation
        corrected_events = {}
        
        for event_type, events in self.curated_events.items():
            corrected_events[event_type] = []
            for onset, _ in events:
                validation = self.timeline_canvas.onset_validations.get(onset, 'pending')
                if validation == 'accepted':
                    corrected_events[event_type].append([onset, 1])
                elif validation == 'rejected':
                    corrected_events[event_type].append([onset, 0])
                else:
                    # Keep pending onsets as is
                    corrected_events[event_type].append([onset, 1])
                    
        # Save to file
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Curated Onsets', '', 'JSON (*.json)')
        if fname:
            with open(fname, 'w') as f:
                json.dump(corrected_events, f, indent=2)
            QMessageBox.information(self, "Success", f"Curated onsets saved to {fname}")
            
    def save_performance_metrics(self):
        metrics = self.performance_metrics
        total = sum(metrics.values())
        
        if total > 0:
            precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
            recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives']) if (metrics['true_positives'] + metrics['false_negatives']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0
            
        results = {
            'performance_metrics': metrics,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_annotated': total
        }
        
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Performance Metrics', '', 'JSON (*.json)')
        if fname:
            with open(fname, 'w') as f:
                json.dump(results, f, indent=2)
            QMessageBox.information(self, "Success", f"Performance metrics saved to {fname}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    annotator = MotionAnnotator()
    annotator.show()
    sys.exit(app.exec_()) 
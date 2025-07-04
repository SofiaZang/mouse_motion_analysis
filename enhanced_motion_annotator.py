import sys
import cv2
import json
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QFileDialog, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QGridLayout, QTextEdit, QComboBox, QCheckBox,
    QMessageBox, QProgressBar, QSplitter, QLineEdit
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
        self.event_offsets = {}  # Store offset frames for events
        
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
        try:
            if self.timeline_line is not None:
                self.timeline_line.remove()
        except Exception:
            pass
        self.timeline_line = self.ax.axvline(self.current_frame, color='red', linewidth=2, alpha=0.8)
        self.draw()
        
    def plot_motion_energy(self, motion_energy, onsets=None, onset_types=None, event_offsets=None, event_status=None):
        import numpy as np
        self.fig.set_size_inches(30, 5)
        self.fig.set_dpi(200)

        self.motion_energy = motion_energy
        self.total_frames = len(motion_energy)
        self.onsets = onsets or []
        self.onset_types = onset_types or {}
        self.event_offsets = event_offsets or {}
        self.event_status = event_status or {}

        self.ax.clear()

        frames = np.arange(self.total_frames)
        motion_signal_norm = (motion_energy - np.min(motion_energy)) / (np.max(motion_energy) - np.min(motion_energy))
        self.ax.plot(frames, motion_signal_norm, color='blue', linewidth=0.3)

        for onset in self.onsets:
            onset_type = self.onset_types.get(onset, 'unknown')
            offset = self.event_offsets.get(onset, onset)
            if onset_type == 'active':
                self.ax.axvspan(onset, offset, color='orange', alpha=0.3)
            elif onset_type == 'twitch':
                self.ax.axvspan(onset, onset+2, color='purple', alpha=0.7)

        # Set limits LAST, after all artists are added
        self.ax.set_xlim(0, max(self.total_frames, 1000))
        self.ax.set_ylim(0, 1.1)
        self.ax.set_ylabel("motion_energy", fontsize=7)
        self.ax.set_xlabel("Frame", fontsize=7)
        self.ax.set_title("classified motion", fontsize=9)
        self.ax.set_yticks([])
        self.ax.tick_params(axis='x', labelsize=6)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.update_timeline()
        
    def set_onset_validation(self, onset, validation):
        self.onset_validations[onset] = validation
        self.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)
        
    def add_event(self, onset, event_type, offset=None):
        """Add a new event to the timeline"""
        self.onsets.append(onset)
        self.onset_types[onset] = event_type
        if offset is not None:
            self.event_offsets[onset] = offset
        self.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)
        
    def remove_event(self, onset):
        """Remove an event from the timeline"""
        if onset in self.onsets:
            self.onsets.remove(onset)
        if onset in self.onset_types:
            del self.onset_types[onset]
        if onset in self.onset_validations:
            del self.onset_validations[onset]
        if onset in self.event_offsets:
            del self.event_offsets[onset]
        self.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)

class MotionAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TwitchCraft")
        self.setGeometry(100, 100, 1400, 900)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        
        # Data storage
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 15
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
        self.fps_spinbox.setRange(1, 50)
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

        # Manual event addition
        manual_event_group = QGroupBox("Manual Event Addition")
        manual_layout = QVBoxLayout()
        event_type_layout = QHBoxLayout()
        event_type_layout.addWidget(QLabel("Event Type:"))
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItems(["twitch", "active", "complex"])
        event_type_layout.addWidget(self.event_type_combo)
        manual_layout.addLayout(event_type_layout)
        onset_offset_layout = QGridLayout()
        self.onset_spinbox = QSpinBox()
        self.onset_spinbox.setRange(0, 999999)
        self.onset_spinbox.valueChanged.connect(self.update_offset_range)
        self.offset_spinbox = QSpinBox()
        self.offset_spinbox.setRange(0, 999999)
        self.set_current_onset_btn = QPushButton("Set Current Frame as Onset")
        self.set_current_onset_btn.clicked.connect(self.set_current_as_onset)
        self.set_current_offset_btn = QPushButton("Set Current Frame as Offset")
        self.set_current_offset_btn.clicked.connect(self.set_current_as_offset)
        onset_offset_layout.addWidget(QLabel("Onset Frame:"), 0, 0)
        onset_offset_layout.addWidget(self.onset_spinbox, 0, 1)
        onset_offset_layout.addWidget(self.set_current_onset_btn, 0, 2)
        onset_offset_layout.addWidget(QLabel("Offset Frame:"), 1, 0)
        onset_offset_layout.addWidget(self.offset_spinbox, 1, 1)
        onset_offset_layout.addWidget(self.set_current_offset_btn, 1, 2)
        manual_layout.addLayout(onset_offset_layout)
        self.add_event_btn = QPushButton("➕ Add Event")
        self.add_event_btn.clicked.connect(self.add_manual_event)
        manual_layout.addWidget(self.add_event_btn)
        manual_event_group.setLayout(manual_layout)
        left_panel.addWidget(manual_event_group)

        # Right panel - Motion energy and annotation
        right_panel = QVBoxLayout()
        timeline_group = QGroupBox("Motion Energy Timeline")
        timeline_layout = QVBoxLayout()
        self.timeline_canvas = DraggableTimeline()
        self.timeline_canvas.timeline_moved.connect(self.timeline_frame_changed)
        timeline_layout.addWidget(self.timeline_canvas)
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
        onset_group = QGroupBox("Onset Navigation & Validation")
        onset_layout = QVBoxLayout()
        nav_layout = QHBoxLayout()
        self.prev_onset_btn = QPushButton("← Prev Onset")
        self.next_onset_btn = QPushButton("Next Onset →")
        self.prev_onset_btn.clicked.connect(self.prev_onset)
        self.next_onset_btn.clicked.connect(self.next_onset)
        nav_layout.addWidget(self.prev_onset_btn)
        nav_layout.addWidget(self.next_onset_btn)
        onset_layout.addLayout(nav_layout)
        self.onset_info_label = QLabel("No onsets loaded")
        onset_layout.addWidget(self.onset_info_label)
        validation_layout = QHBoxLayout()
        self.accept_btn = QPushButton("✓ Accept")
        self.reject_btn = QPushButton("✗ Reject")
        self.delete_btn = QPushButton("🗑 Delete")
        self.accept_btn.clicked.connect(lambda: self.validate_onset('accepted'))
        self.reject_btn.clicked.connect(lambda: self.validate_onset('rejected'))
        self.delete_btn.clicked.connect(self.delete_current_onset)
        validation_layout.addWidget(self.accept_btn)
        validation_layout.addWidget(self.reject_btn)
        validation_layout.addWidget(self.delete_btn)
        onset_layout.addLayout(validation_layout)
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
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(100)
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        right_panel.addWidget(metrics_group)
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        splitter.addWidget(left_widget)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        splitter.setSizes([700, 900])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
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
            
            # Update onset/offset ranges
            self.onset_spinbox.setMaximum(self.total_frames - 1)
            self.offset_spinbox.setMaximum(self.total_frames - 1)
            
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
                
            # Pad to divisible by 5 and average
            self.prepare_motion_energy()
            self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, self.timeline_canvas.event_status)
            
    def prepare_motion_energy(self):
        """Pad motion energy to divisible by 5 and average"""
        if self.motion_energy is None:
            return
        # Ensure numpy float64 array
        self.motion_energy = np.asarray(self.motion_energy, dtype=np.float64)
        # Pad to make divisible by 5
        remainder = len(self.motion_energy) % 5
        if remainder != 0:
            padding_needed = 5 - remainder
            self.motion_energy = np.pad(self.motion_energy, (0, padding_needed), mode='edge')
        # Reshape and average
        new_length = len(self.motion_energy) // 5
        self.motion_energy = self.motion_energy.reshape(new_length, 5).mean(axis=1)
        # Update total frames
        self.total_frames = len(self.motion_energy)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.onset_spinbox.setMaximum(self.total_frames - 1)
        self.offset_spinbox.setMaximum(self.total_frames - 1)
            
    def load_classifications(self):
        try:
            fname, _ = QFileDialog.getOpenFileName(self, 'Load Classifications', '', 'JSON/CSV/Excel (*.json *.csv *.xlsx)')
            if fname:
                if fname.endswith('.json'):
                    self.load_json_classifications(fname)
                else:
                    import pandas as pd
                    df = pd.read_csv(fname) if fname.endswith('.csv') else pd.read_excel(fname)
                    cols = set(df.columns)
                    if {'active', 'twitch'}.issubset(cols):
                        self.load_framewise_table(fname)
                    elif {'active_motion_onset', 'active_motion_offset', 'twitch_onset', 'twitch_offset'}.issubset(cols):
                        self.load_excel_classifications(fname)
                    else:
                        QMessageBox.warning(self, "Warning", "File format not recognized. Please provide a valid classification file.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load classifications: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def load_json_classifications(self, fname):
        """Load classifications from JSON format"""
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
        self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, self.timeline_canvas.event_status)
        
        # Go to first onset
        if self.onsets:
            self.goto_onset(0)
            
    def load_excel_classifications(self, fname):
        """Load classifications from Excel/CSV format with SLEAP-like structure"""
        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(fname)
            else:
                df = pd.read_excel(fname)
                
            print(f"Loaded CSV with shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Check if this is the expected format
            expected_columns = ['frame_idx', 'motion_energy', 'active_motion_onset', 'active_motion_offset', 
                               'twitch_onset', 'twitch_offset', 'complex_motion_onset', 'complex_motion_offset']
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                QMessageBox.warning(self, "Warning", f"Missing columns: {missing_columns}\nExpected: {expected_columns}")
                return
                
            # Load motion energy if not already loaded
            if self.motion_energy is None:
                self.motion_energy = df['motion_energy'].values.astype(np.float64)
                self.prepare_motion_energy()
            else:
                # Check if motion energy lengths match
                csv_length = len(df)
                me_length = len(self.motion_energy)
                if csv_length != me_length:
                    QMessageBox.warning(self, "Warning", f"Motion energy length mismatch: CSV has {csv_length} frames, loaded motion energy has {me_length} frames")
                    return
                
            # Extract events from onset/offset arrays
            self.onsets = []
            self.onset_types = {}
            self.timeline_canvas.event_offsets = {}
            self.event_status = {}
            
            # Process active motion events
            active_onsets = np.where(df['active_motion_onset'] == 1)[0]
            active_offsets = np.where(df['active_motion_offset'] == 1)[0]
            print(f"Found {len(active_onsets)} active onsets and {len(active_offsets)} active offsets")
            print(f"Active onsets: {active_onsets[:10]}...")  # Show first 10
            
            for onset in active_onsets:
                # Find corresponding offset
                offset = self.find_corresponding_offset(onset, active_offsets)
                self.onsets.append(onset)
                self.onset_types[onset] = 'active'
                self.timeline_canvas.event_offsets[onset] = offset
                self.event_status[onset] = 1  # default to accepted for now
                print(f"Added active event: onset={onset}, offset={offset}")
                
            # Process twitch events
            twitch_onsets = np.where(df['twitch_onset'] == 1)[0]
            twitch_offsets = np.where(df['twitch_offset'] == 1)[0]
            print(f"Found {len(twitch_onsets)} twitch onsets and {len(twitch_offsets)} twitch offsets")
            print(f"Twitch onsets: {twitch_onsets[:10]}...")  # Show first 10
            
            for onset in twitch_onsets:
                # Find corresponding offset
                offset = self.find_corresponding_offset(onset, twitch_offsets)
                self.onsets.append(onset)
                self.onset_types[onset] = 'twitch'
                self.timeline_canvas.event_offsets[onset] = offset
                self.event_status[onset] = 1
                print(f"Added twitch event: onset={onset}, offset={offset}")
                
            # Process complex motion events
            complex_onsets = np.where(df['complex_motion_onset'] == 1)[0]
            complex_offsets = np.where(df['complex_motion_offset'] == 1)[0]
            print(f"Found {len(complex_onsets)} complex onsets and {len(complex_offsets)} complex offsets")
            
            for onset in complex_onsets:
                # Find corresponding offset
                offset = self.find_corresponding_offset(onset, complex_offsets)
                self.onsets.append(onset)
                self.onset_types[onset] = 'complex'
                self.timeline_canvas.event_offsets[onset] = offset
                
            # Sort onsets
            self.onsets = sorted(self.onsets)
            self.current_onset_idx = 0
            
            print(f"Total events loaded: {len(self.onsets)}")
            print(f"Event types: {set(self.onset_types.values())}")
            print(f"Onset range: {min(self.onsets) if self.onsets else 'N/A'} to {max(self.onsets) if self.onsets else 'N/A'}")
            print(f"Motion energy length: {len(self.motion_energy)}")
            
            # Check for out-of-range onsets
            if self.onsets:
                max_onset = max(self.onsets)
                if max_onset >= len(self.motion_energy):
                    QMessageBox.warning(self, "Warning", f"Some onsets ({max_onset}) are beyond motion energy length ({len(self.motion_energy)})")
                    # Filter out out-of-range onsets
                    valid_onsets = [onset for onset in self.onsets if onset < len(self.motion_energy)]
                    self.onsets = valid_onsets
                    print(f"Filtered to {len(self.onsets)} valid onsets")
            
            # Create curated events structure
            self.curated_events = {}
            for onset in self.onsets:
                event_type = self.onset_types[onset]
                if event_type not in self.curated_events:
                    self.curated_events[event_type] = []
                self.curated_events[event_type].append([onset, 1])  # 1 indicates detected by algorithm
                
            # Update timeline
            self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, self.event_status)
            
            # Go to first onset
            if self.onsets:
                self.goto_onset(0)
                
            QMessageBox.information(self, "Success", f"Loaded {len(self.onsets)} events from {fname}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")
            print(f"Error loading CSV: {e}")
            import traceback
            traceback.print_exc()
        
    def find_corresponding_offset(self, onset, offset_indices):
        """Find the corresponding offset for a given onset"""
        # Find the next offset after the onset
        next_offsets = offset_indices[offset_indices > onset]
        if len(next_offsets) > 0:
            return next_offsets[0]
        else:
            # If no offset found, use onset + 1 frame
            return onset + 1
        
    def set_current_as_onset(self):
        """Set current frame as onset"""
        self.onset_spinbox.setValue(self.current_frame)
        
    def set_current_as_offset(self):
        """Set current frame as offset"""
        self.offset_spinbox.setValue(self.current_frame)
        
    def update_offset_range(self):
        """Update offset range to be >= onset"""
        onset = self.onset_spinbox.value()
        self.offset_spinbox.setMinimum(onset)
        if self.offset_spinbox.value() < onset:
            self.offset_spinbox.setValue(onset)
            
    def add_manual_event(self):
        """Add a manually specified event"""
        onset = self.onset_spinbox.value()
        offset = self.offset_spinbox.value()
        event_type = self.event_type_combo.currentText()
        
        if onset >= offset:
            QMessageBox.warning(self, "Warning", "Onset must be less than offset")
            return
            
        # Add to timeline
        self.timeline_canvas.add_event(onset, event_type, offset)
        
        # Add to data structures
        self.onsets.append(onset)
        self.onset_types[onset] = event_type
        
        # Sort onsets
        self.onsets = sorted(self.onsets)
        
        # Update curated events
        if event_type not in self.curated_events:
            self.curated_events[event_type] = []
        self.curated_events[event_type].append([onset, 1])  # 1 indicates valid event
        
        # Go to the new event
        new_idx = self.onsets.index(onset)
        self.goto_onset(new_idx)
        
        QMessageBox.information(self, "Success", f"Added {event_type} event from frame {onset} to {offset}")
        
    def delete_current_onset(self):
        """Delete the currently selected onset"""
        if not self.onsets:
            return
            
        current_onset = self.onsets[self.current_onset_idx]
        
        # Remove from timeline
        self.timeline_canvas.remove_event(current_onset)
        
        # Remove from data structures
        self.onsets.remove(current_onset)
        if current_onset in self.onset_types:
            del self.onset_types[current_onset]
            
        # Remove from curated events
        for event_type, events in self.curated_events.items():
            self.curated_events[event_type] = [event for event in events if event[0] != current_onset]
            
        # Update current index
        if self.onsets:
            if self.current_onset_idx >= len(self.onsets):
                self.current_onset_idx = len(self.onsets) - 1
            self.goto_onset(self.current_onset_idx)
        else:
            self.current_onset_idx = 0
            self.update_onset_info()
            
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
        offset = self.timeline_canvas.event_offsets.get(current_onset, current_onset)
        
        info_text = f"Onset {self.current_onset_idx + 1}/{len(self.onsets)}: Frame {current_onset}"
        if offset != current_onset:
            info_text += f" to {offset}"
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
            
    def shift_onset(self, shift):
        if not self.onsets:
            return
        current_onset = self.onsets[self.current_onset_idx]
        new_frame = max(0, min(self.total_frames - 1, current_onset + shift))
        old_onset = current_onset
        self.onsets[self.current_onset_idx] = new_frame

        # Update in curated events
        for event_type, events in self.curated_events.items():
            for event in events:
                if event[0] == old_onset:
                    event[0] = new_frame
                    self.onset_types[new_frame] = self.onset_types.pop(old_onset, event_type)
                    # Mark as edited if event_status exists
                    if hasattr(self, 'event_status'):
                        self.event_status[new_frame] = 0.5
                    break

        # Redraw plot and update UI
        self.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))
        self.goto_onset(self.onsets.index(new_frame))

    def update_performance_display(self):
        metrics = self.performance_metrics
        total = metrics['true_positives'] + metrics['false_positives']
        if total > 0:
            precision = metrics['true_positives'] / total if total > 0 else 0
        else:
            precision = 0
        display_text = f"""
Performance Metrics:
- True Positives: {metrics['true_positives']}
- False Positives: {metrics['false_positives']}
- Precision: {precision:.3f}
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
                offset = self.timeline_canvas.event_offsets.get(onset, onset)
                
                if validation == 'accepted':
                    corrected_events[event_type].append([onset, offset, 1])  # Include offset
                elif validation == 'rejected':
                    corrected_events[event_type].append([onset, offset, 0])
                else:
                    # Keep pending onsets as is
                    corrected_events[event_type].append([onset, offset, 1])
                    
        # Save to file
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Curated Onsets', '', 'JSON (*.json)')
        if fname:
            with open(fname, 'w') as f:
                json.dump(corrected_events, f, indent=2)
            QMessageBox.information(self, "Success", f"Curated onsets saved to {fname}")
            
    def save_performance_metrics(self):
        metrics = self.performance_metrics
        total = metrics['true_positives'] + metrics['false_positives']
        if total > 0:
            precision = metrics['true_positives'] / total if total > 0 else 0
        else:
            precision = 0
        results = {
            'performance_metrics': metrics,
            'precision': precision,
            'total_annotated': total
        }
        
        fname, _ = QFileDialog.getSaveFileName(self, 'Save Performance Metrics', '', 'JSON (*.json)')
        if fname:
            with open(fname, 'w') as f:
                json.dump(results, f, indent=2)
            QMessageBox.information(self, "Success", f"Performance metrics saved to {fname}")

    def load_framewise_table(self, fname):
        import pandas as pd
        df = pd.read_csv(fname) if fname.endswith('.csv') else pd.read_excel(fname)
        self.motion_energy = df['motion_energy'].values
        self.onsets = []
        self.onset_types = {}
        self.timeline_canvas.event_offsets = {}
        self.event_status = {}
        self.curated_events = {}

        for event_type in ['active', 'twitch']:
            arr = df[event_type].values
            in_event = False
            for i, val in enumerate(arr):
                if val == 1 and not in_event:
                    onset = i
                    in_event = True
                elif val == 0 and in_event:
                    offset = i
                    self.onsets.append(onset)
                    self.onset_types[onset] = event_type
                    self.timeline_canvas.event_offsets[onset] = offset
                    self.event_status[onset] = 1  # default to accepted for now
                    if event_type not in self.curated_events:
                        self.curated_events[event_type] = []
                    self.curated_events[event_type].append([onset, offset, 1])
                    in_event = False
            if in_event:
                offset = len(arr)
                self.onsets.append(onset)
                self.onset_types[onset] = event_type
                self.timeline_canvas.event_offsets[onset] = offset
                self.event_status[onset] = 1
                if event_type not in self.curated_events:
                    self.curated_events[event_type] = []
                self.curated_events[event_type].append([onset, offset, 1])

        self.onsets = sorted(self.onsets)
        self.current_onset_idx = 0
        self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, self.event_status)
        if self.onsets:
            self.goto_onset(0)

    # Allow moving the movie with left/right arrow keys and by dragging the timeline
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            if self.current_frame < self.total_frames - 1:
                self.current_frame += 1
                self.frame_slider.setValue(self.current_frame)
                self.show_frame(self.current_frame)
                self.timeline_canvas.current_frame = self.current_frame
                self.timeline_canvas.update_timeline()
        elif event.key() == Qt.Key_Left:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.frame_slider.setValue(self.current_frame)
                self.show_frame(self.current_frame)
                self.timeline_canvas.current_frame = self.current_frame
                self.timeline_canvas.update_timeline()
        elif event.key() == Qt.Key_Space:
            if self.timer.isActive():
                self.pause()
            else:
                self.play()
        else:
            super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    annotator = MotionAnnotator()
    annotator.show()
    sys.exit(app.exec_()) 
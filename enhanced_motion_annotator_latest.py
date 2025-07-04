import sys
import cv2
import json
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QFileDialog, QSpinBox, QDoubleSpinBox, 
    QGroupBox, QGridLayout, QTextEdit, QComboBox, QCheckBox,
    QMessageBox, QProgressBar, QSplitter, QLineEdit, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIntValidator, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PyQt5.QtWidgets import QStyle
import csv

class DraggableTimeline(FigureCanvas):
    """Custom matplotlib canvas with draggable timeline"""
    timeline_moved = pyqtSignal(int)
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4.0))  # Increased height for better visibility
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

        self.motion_energy = motion_energy
        self.total_frames = len(motion_energy)
        self.onsets = onsets or []
        self.onset_types = onset_types or {}
        self.event_offsets = event_offsets or {}
        self.event_status = event_status or {}

        self.ax.clear()
        self.ax.plot(np.arange(self.total_frames), self.motion_energy, color='blue', linewidth=1)

        # Plot onset-to-offset spans
        for onset in self.onsets:
            onset_type = self.onset_types.get(onset, '')
            offset = self.event_offsets.get(onset, onset)
            status = self.event_status.get(onset, '')
            alpha = 1.0 if status == 'accepted' else 0.4
            
            if onset_type == 'twitch':
                self.ax.axvspan(onset, offset, color='purple', alpha=alpha)
            elif onset_type == 'active':
                self.ax.axvspan(onset, offset, color='yellow', alpha=alpha)

        # Add validation markers as bullet points on top
        for onset in self.onsets:
            validation = self.onset_validations.get(onset, 'pending')
            if validation == 'accepted':
                # Green bullet for accepted
                self.ax.plot(onset, max(self.motion_energy) * 1.05, 'o', color='green', markersize=6)
            elif validation == 'rejected':
                # Red bullet for rejected
                self.ax.plot(onset, max(self.motion_energy) * 1.05, 'o', color='red', markersize=6)
            elif validation == 'edited':
                # Orange bullet for edited
                self.ax.plot(onset, max(self.motion_energy) * 1.05, 'o', color='orange', markersize=6)

        self.ax.set_xlim(0, max(self.total_frames, 1000))
        self.ax.set_ylim(0, max(self.motion_energy) * 1.2)  # Increased ylim to accommodate bullets
        self.ax.set_ylabel("motion_energy", fontsize=7)
        self.ax.set_xlabel("Frame", fontsize=7)
        self.ax.set_title("classified motion", fontsize=9)
        self.ax.set_yticks([])
        self.ax.tick_params(axis='x', labelsize=6)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.update_timeline()
        
    def plot_motion_energy_preserve_view(self, *args, **kwargs):
        self.plot_motion_energy(*args, **kwargs)
        self.draw()

    def set_onset_validation(self, onset, validation):
        # Store current view limits
        # xlim = self.ax.get_xlim()
        # ylim = self.ax.get_ylim()
        self.onset_validations[onset] = validation
        self.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)
        # Restore view limits
        # self.ax.set_xlim(xlim)
        # self.ax.set_ylim(ylim)
        self.draw()
        
    def add_event(self, onset, event_type, offset=None):
        """Add a new event to the timeline"""
        self.onsets.append(onset)
        self.onset_types[onset] = event_type
        if offset is not None:
            self.event_offsets[onset] = offset
        self.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)
        
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
        self.plot_motion_energy_preserve_view(self.motion_energy, self.onsets, self.onset_types, self.event_offsets, self.event_status)

    def wheelEvent(self, event):
        # Zoom in/out on scroll, centered on current_frame
        ax = self.ax
        xlim = ax.get_xlim()
        center = self.current_frame
        width = xlim[1] - xlim[0]
        if event.angleDelta().y() > 0:
            # Zoom in
            new_width = width / 2
        else:
            # Zoom out
            new_width = width * 2
        new_xlim = (max(center - new_width/2, 0), min(center + new_width/2, self.total_frames))
        ax.set_xlim(new_xlim)
        self.draw()
        event.accept()

class MotionAnnotator(QWidget):
    """J'ai rajouté cette fonction"""
    def find_closest_onset_idx(self, frame):
        """Return the index of the closest onset <= frame, or 0 if none."""
        if not self.onsets:
            return 0
        idx = 0
        for i, onset in enumerate(self.onsets):
            if onset <= frame:
                idx = i
            else:
                break
        return idx

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
        }
        
        self.init_ui()
        self.setup_timer()
        
    def init_ui(self):
        # Main layout
        main_layout = QHBoxLayout()

        # Left panel - Video and controls
        left_panel = QVBoxLayout()

        # Video display - smaller size with zoom capability
        self.video_label = QLabel("Load a video to start")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 450)  # Made smaller
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Make it expand
        self.video_label.setStyleSheet("border: 2px solid gray;")
        self.video_zoom_factor = 1.0  # Track zoom level
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
        # Remove speed and fps spinboxes, add FPS QLineEdit
        self.fps_lineedit = QLineEdit()
        self.fps_lineedit.setPlaceholderText("FPS")
        self.fps_lineedit.setValidator(QIntValidator(1, 1000, self))
        self.fps_lineedit.textChanged.connect(self.handle_fps_change)
        # Load video button on its own row
        video_layout.addWidget(self.load_video_btn, 0, 0, 1, 4)  # Span all 4 columns
        
        # Play/pause/stop buttons on second row with equal sizes, next to each other
        self.play_btn.setFixedWidth(80)
        self.pause_btn.setFixedWidth(80)
        self.stop_btn.setFixedWidth(80)
        play_pause_layout = QHBoxLayout()
        play_pause_layout.addWidget(self.play_btn)
        play_pause_layout.addWidget(self.pause_btn)
        play_pause_layout.addWidget(self.stop_btn)
        video_layout.addLayout(play_pause_layout, 1, 0, 1, 4)  # Span all columns
        
        # FPS controls on third row
        self.fps_lineedit.setFixedWidth(60)  # Make FPS input smaller
        video_layout.addWidget(QLabel("FPS:"), 2, 0)
        video_layout.addWidget(self.fps_lineedit, 2, 1)
        
        # Add video zoom controls
        self.zoom_in_video_btn = QPushButton("🔍+")
        self.zoom_out_video_btn = QPushButton("🔍-")
        self.reset_video_zoom_btn = QPushButton("Reset Zoom")
        
        # Set fixed width for zoom buttons to make them the same size
        self.zoom_in_video_btn.setFixedWidth(70)
        self.zoom_out_video_btn.setFixedWidth(70)
        
        self.zoom_in_video_btn.clicked.connect(self.zoom_in_video)
        self.zoom_out_video_btn.clicked.connect(self.zoom_out_video)
        self.reset_video_zoom_btn.clicked.connect(self.reset_video_zoom)
        
        # Create horizontal layout for zoom buttons
        zoom_buttons_layout = QHBoxLayout()
        zoom_buttons_layout.addWidget(self.zoom_in_video_btn)
        zoom_buttons_layout.addWidget(self.zoom_out_video_btn)
        zoom_buttons_layout.addWidget(self.reset_video_zoom_btn)
        
        # Make reset zoom button fit with zoom buttons
        self.reset_video_zoom_btn.setFixedWidth(80)
        
        video_layout.addWidget(QLabel("Video Zoom:"), 3, 0)
        video_layout.addLayout(zoom_buttons_layout, 3, 1)
        
        video_controls.setLayout(video_layout)
        left_panel.addWidget(video_controls)

        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_moved)
        left_panel.addWidget(self.frame_slider)

        # Frame info
        self.frame_info_label = QLabel("Frame: 0 / 0")
        left_panel.addWidget(self.frame_info_label)
        
        # Add onset status bar below video
        self.onset_status_label = QLabel("No onset at current frame")
        self.onset_status_label.setStyleSheet("background-color: lightgray; padding: 5px; border: 1px solid gray;")
        left_panel.addWidget(self.onset_status_label)

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
        self.timeline_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow more vertical expansion
        self.timeline_canvas.setMinimumHeight(350)  # Increased minimum height
        self.timeline_canvas.setMaximumHeight(500)  # Increased maximum height
        self.timeline_canvas.timeline_moved.connect(self.timeline_frame_changed)
        timeline_layout.addWidget(self.timeline_canvas)
        
        # Add custom zoom in/out and reset buttons (same style as video zoom)
        zoom_btn_layout = QHBoxLayout()
        self.zoom_in_btn = QPushButton('🔍+')
        self.zoom_out_btn = QPushButton('🔍-')
        self.reset_zoom_btn = QPushButton('Reset Zoom')
        self.zoom_in_btn.clicked.connect(self.zoom_in_timeline)
        self.zoom_out_btn.clicked.connect(self.zoom_out_timeline)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom_timeline)
        zoom_btn_layout.addWidget(self.zoom_in_btn)
        zoom_btn_layout.addWidget(self.zoom_out_btn)
        zoom_btn_layout.addWidget(self.reset_zoom_btn)
        timeline_layout.addLayout(zoom_btn_layout)
        
        timeline_controls = QHBoxLayout()
        self.load_me_btn = QPushButton("Load Motion Energy")
        self.load_me_btn.clicked.connect(self.load_motion_energy)
        self.load_class_btn = QPushButton("Load Classifications")
        self.load_class_btn.clicked.connect(self.load_classifications)
        timeline_controls.addWidget(self.load_me_btn)
        timeline_controls.addWidget(self.load_class_btn)
        timeline_layout.addLayout(timeline_controls)
        timeline_group.setLayout(timeline_layout)
        onset_group = QGroupBox("Onset Navigation & Validation")
        onset_layout = QVBoxLayout()
        # Add filter combo boxes side by side
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Event type"))
        self.onset_filter_combo = QComboBox()
        self.onset_filter_combo.addItems(["All", "Active", "Twitch"])
        self.onset_filter_combo.currentIndexChanged.connect(self.update_onset_filter)
        filter_layout.addWidget(self.onset_filter_combo)
        filter_layout.addWidget(QLabel("Event status"))
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItems(["All", "Editing", "Pending", "Accepted", "Rejected", "Manually Added"])
        self.status_filter_combo.currentIndexChanged.connect(self.update_onset_filter)
        filter_layout.addWidget(self.status_filter_combo)
        onset_layout.addLayout(filter_layout)
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
        self.accept_btn.clicked.connect(lambda: self.validate_onset('accepted'))
        self.reject_btn.clicked.connect(lambda: self.validate_onset('rejected'))
        validation_layout.addWidget(self.accept_btn)
        validation_layout.addWidget(self.reject_btn)
        # Add Edit button
        self.edit_btn = QPushButton("✎ Edit")
        self.edit_btn.clicked.connect(self.start_edit_onset)
        validation_layout.addWidget(self.edit_btn)
        onset_layout.addLayout(validation_layout)
        # Remove shift left/right buttons and their layout
        # Add edit widgets (hidden by default)
        self.edit_widget = QWidget()
        edit_form = QHBoxLayout()
        self.edit_onset_spinbox = QSpinBox()
        self.edit_onset_spinbox.setRange(0, 999999)
        self.edit_offset_spinbox = QSpinBox()
        self.edit_offset_spinbox.setRange(0, 999999)
        edit_form.addWidget(QLabel("New Onset:"))
        edit_form.addWidget(self.edit_onset_spinbox)
        edit_form.addWidget(QLabel("New Offset:"))
        edit_form.addWidget(self.edit_offset_spinbox)
        self.finish_edit_btn = QPushButton("Finish Edit")
        self.finish_edit_btn.clicked.connect(self.finish_edit_onset)
        edit_form.addWidget(self.finish_edit_btn)
        self.edit_widget.setLayout(edit_form)
        self.edit_widget.hide()
        onset_layout.addWidget(self.edit_widget)
        onset_group.setLayout(onset_layout)
        save_group = QGroupBox("Save & Export")
        save_layout = QVBoxLayout()
        self.save_export_btn = QPushButton("💾 Save and Export Validation")
        self.save_export_btn.setStyleSheet("QPushButton { font-weight: bold; font-size: 12px; padding: 8px; }")
        self.save_export_btn.clicked.connect(self.save_and_export_validation)
        save_layout.addWidget(self.save_export_btn)
        save_group.setLayout(save_layout)
        metrics_group = QGroupBox("Performance Score")
        metrics_layout = QVBoxLayout()
        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(50)
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        right_panel.addWidget(timeline_group)
        right_panel.addWidget(onset_group)
        right_panel.addWidget(save_group)
        right_panel.addWidget(metrics_group)
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        splitter.addWidget(left_widget)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        splitter.addWidget(right_widget)
        splitter.setSizes([850, 750])  # More balanced - give motion energy more space
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
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.fps_lineedit.setText("")  # Clear FPS input, user must enter manually
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
        self.timeline_canvas.event_offsets[onset] = offset  # Store the offset
        
        # Sort onsets
        self.onsets = sorted(self.onsets)
        
        # Update curated events
        if event_type not in self.curated_events:
            self.curated_events[event_type] = []
        self.curated_events[event_type].append([onset, offset, 0])  # Store onset, offset, and validation status (0 = manually added)
        
        # Set validation status to manually added
        self.timeline_canvas.onset_validations[onset] = 'manually added'
        
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
            # Only play if FPS is set
            if hasattr(self, 'fps') and self.fps >= 1:
                interval = int(1000 / self.fps)
                self.timer.start(interval)
            else:
                QMessageBox.warning(self, "Warning", "Please enter a valid FPS before playing.")
            
    def pause(self):
        self.timer.stop()
        
    def stop(self):
        self.timer.stop()
        self.current_frame = 0
        self.frame_slider.setValue(0)
        self.show_frame(0)
        
    def handle_fps_change(self):
        text = self.fps_lineedit.text()
        if text.isdigit() and int(text) >= 1:
            self.fps = int(text)
            # Update timer interval if playing
            if hasattr(self, 'timer') and self.timer.isActive():
                interval = int(1000 / self.fps)
                self.timer.start(interval)
        # If invalid, do not update self.fps
            
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
            
        # Check if current frame is an onset (for status bar update)
        onset_type = self.onset_types.get(frame_num, None)
        
        # Optimize video processing for better performance
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Use faster scaling for better performance with zoom support
        pixmap = QPixmap.fromImage(q_img)
        label_size = self.video_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            # Apply zoom factor
            zoomed_width = int(label_size.width() * self.video_zoom_factor)
            zoomed_height = int(label_size.height() * self.video_zoom_factor)
            zoomed_size = QSize(zoomed_width, zoomed_height)
            scaled_pixmap = pixmap.scaled(zoomed_size, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        
        self.update_frame_info()
        self.update_onset_status()
        
    def slider_moved(self, value):
        self.current_frame = value
        self.show_frame(value)
        self.timeline_canvas.current_frame = value
        self.timeline_canvas.update_timeline()
        # Auto-select closest onset
        if self.onsets:
            idx = self.find_closest_onset_idx(value)
            if idx != self.current_onset_idx:
                self.goto_onset(idx)
            else:
                self.update_onset_info()

    def timeline_frame_changed(self, frame):
        self.current_frame = frame
        self.frame_slider.setValue(frame)
        self.show_frame(frame)
        # Auto-select closest onset
        if self.onsets:
            idx = self.find_closest_onset_idx(frame)
            if idx != self.current_onset_idx:
                self.goto_onset(idx)
            else:
                self.update_onset_info()
        
    def update_frame_info(self):
        self.frame_info_label.setText(f"Frame: {self.current_frame} / {self.total_frames}")
        
    def update_onset_filter(self):
        # Update filtered_onsets based on both filter selections
        type_text = self.onset_filter_combo.currentText().lower()
        status_text = self.status_filter_combo.currentText().lower()
        # Filter by type
        if type_text == "all":
            filtered = self.onsets.copy()
        else:
            filtered = [o for o in self.onsets if self.onset_types.get(o, '').lower() == type_text]
        # Filter by status
        if status_text != "all":
            # Handle the "Manually Added" case specifically
            if status_text == "manually added":
                filtered = [o for o in filtered if self.timeline_canvas.onset_validations.get(o, 'pending') == 'manually added']
            else:
                filtered = [o for o in filtered if self.timeline_canvas.onset_validations.get(o, 'pending').lower() == status_text]
        self.filtered_onsets = filtered
        # Reset current_onset_idx to 0 if needed
        if not self.filtered_onsets:
            self.current_onset_idx = 0
        else:
            # Try to keep the current frame in view if possible
            current_frame = self.current_frame
            idx = 0
            for i, onset in enumerate(self.filtered_onsets):
                if onset <= current_frame:
                    idx = i
                else:
                    break
            self.current_onset_idx = idx
        self.update_onset_info()
        
    def goto_onset(self, idx):
        # Use filtered_onsets for navigation
        if not hasattr(self, 'filtered_onsets') or not self.filtered_onsets:
            self.filtered_onsets = self.onsets.copy()
        if 0 <= idx < len(self.filtered_onsets):
            self.current_onset_idx = idx
            onset_frame = self.filtered_onsets[idx]
            self.current_frame = onset_frame
            self.frame_slider.setValue(onset_frame)
            self.show_frame(onset_frame)
            self.timeline_canvas.current_frame = onset_frame
            self.timeline_canvas.update_timeline()
            self.update_onset_info()
            
    def prev_onset(self):
        if not hasattr(self, 'filtered_onsets') or not self.filtered_onsets:
            self.filtered_onsets = self.onsets.copy()
        if self.current_onset_idx > 0:
            self.goto_onset(self.current_onset_idx - 1)
            
    def next_onset(self):
        if not hasattr(self, 'filtered_onsets') or not self.filtered_onsets:
            self.filtered_onsets = self.onsets.copy()
        if self.current_onset_idx < len(self.filtered_onsets) - 1:
            self.goto_onset(self.current_onset_idx + 1)
            
    def update_onset_info(self):
        if not hasattr(self, 'filtered_onsets') or not self.filtered_onsets:
            self.filtered_onsets = self.onsets.copy()
        if not self.filtered_onsets:
            self.onset_info_label.setText("No onsets loaded")
            return
        current_onset = self.filtered_onsets[self.current_onset_idx]
        onset_type = self.onset_types.get(current_onset, 'unknown')
        validation = self.timeline_canvas.onset_validations.get(current_onset, 'pending')
        offset = self.timeline_canvas.event_offsets.get(current_onset, current_onset)
        info_text = f"Onset {self.current_onset_idx + 1}/{len(self.filtered_onsets)}: Frame {current_onset}"
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
            
    def start_edit_onset(self):
        # Show edit widget and prefill with current onset/offset
        if not self.onsets:
            return
        current_onset = self.onsets[self.current_onset_idx]
        offset = self.timeline_canvas.event_offsets.get(current_onset, current_onset)
        self.edit_onset_spinbox.setValue(current_onset)
        self.edit_offset_spinbox.setValue(offset)
        self.edit_widget.show()
    def finish_edit_onset(self):
        # Update onset/offset, set validation to 'edited', update plot and status
        if not self.onsets:
            return
        new_onset = self.edit_onset_spinbox.value()
        new_offset = self.edit_offset_spinbox.value()
        old_onset = self.onsets[self.current_onset_idx]
        # Remove old onset and add new one
        self.onsets[self.current_onset_idx] = new_onset
        self.onset_types[new_onset] = self.onset_types.pop(old_onset, 'unknown')
        self.timeline_canvas.event_offsets[new_onset] = new_offset
        if old_onset in self.timeline_canvas.event_offsets:
            del self.timeline_canvas.event_offsets[old_onset]
        # Update curated_events
        for event_type, events in self.curated_events.items():
            for event in events:
                if event[0] == old_onset:
                    event[0] = new_onset
                    event[1] = new_offset
        # Set validation to 'edited'
        self.timeline_canvas.onset_validations[new_onset] = 'edited'
        if old_onset in self.timeline_canvas.onset_validations:
            del self.timeline_canvas.onset_validations[old_onset]
        # Set performance to 0.5
        if hasattr(self, 'event_status'):
            self.event_status[new_onset] = 0.5
            if old_onset in self.event_status:
                del self.event_status[old_onset]
        # Hide edit widget
        self.edit_widget.hide()
        # Sort onsets and update index
        self.onsets = sorted(self.onsets)
        self.current_onset_idx = self.onsets.index(new_onset)
        # Redraw plot and update UI
        self.timeline_canvas.plot_motion_energy(self.motion_energy, self.onsets, self.onset_types, self.timeline_canvas.event_offsets, getattr(self, 'event_status', None))
        self.update_onset_info()
        self.update_performance_display()

    def update_performance_display(self):
        # New: Score is 1 for accepted, -1 for rejected, 0.5 for edited, 0 for pending/manually added
        scores = []
        for onset in self.onsets:
            status = self.timeline_canvas.onset_validations.get(onset, 'pending')
            if status == 'accepted':
                scores.append(1)
            elif status == 'rejected':
                scores.append(-1)
            elif status == 'edited':
                scores.append(0.5)
            elif status == 'manually added':
                scores.append(0)
            else:
                scores.append(0)
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0
        display_text = f"""
Performance Score:
- Average Score: {avg_score:.3f}
(accepted=1, rejected=-1, edited=0.5, pending=0, manually added=0)
        """
        self.metrics_text.setText(display_text.strip())
        
    # Removed save_curated_onsets() - functionality merged into save_and_export_validation()
            
    def save_and_export_validation(self):
        """Combined function to save validation data, performance metrics, and create comparison plot"""
        if not self.curated_events:
            QMessageBox.warning(self, "Warning", "No onsets to save")
            return

        # Create a flat list of events with 5 columns each
        export_events = []
        for event_type, events in self.curated_events.items():
            for event in events:
                onset = event[0]
                offset = event[1] if len(event) > 1 else onset
                status = self.timeline_canvas.onset_validations.get(onset, 'pending')
                if status == 'accepted':
                    score = 1
                elif status == 'rejected':
                    score = -1
                elif status == 'edited':
                    score = 0.5
                elif status == 'manually added':
                    score = 0
                else:
                    score = 0
                export_events.append([onset, offset, event_type, status, score])

        # Save to CSV file
        fname, _ = QFileDialog.getSaveFileName(self, 'Save and Export Validation', '', 'CSV (*.csv)')
        if fname:
            if not fname.endswith('.csv'):
                fname += '.csv'
            
            # Create twitchcraft output directory
            import os
            base_dir = os.path.dirname(fname)
            output_dir = os.path.join(base_dir, "twitchcraft_output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(fname))[0]
            
            # Save CSV in output directory
            csv_path = os.path.join(output_dir, f"{base_name}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['onset', 'offset', 'event_type', 'status', 'score'])
                writer.writerows(export_events)
            
            # Save numpy file in output directory
            np_path = os.path.join(output_dir, f"{base_name}.npy")
            np.save(np_path, np.array(export_events, dtype=object))
            
            # Save performance metrics as JSON in output directory
            metrics_path = os.path.join(output_dir, f"{base_name}_metrics.json")
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
            with open(metrics_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create comparison plot and save it in output directory
            plot_path = os.path.join(output_dir, f"{base_name}_comparison_plot.png")
            self.create_validation_comparison_plot(export_events, save_path=plot_path)
            
            # Show success message with all saved files
            QMessageBox.information(self, "Success", 
                f"Validation exported successfully!\n\n"
                f"Files saved in 'twitchcraft_output' directory:\n"
                f"• {base_name}.csv (CSV data)\n"
                f"• {base_name}.npy (NumPy data)\n"
                f"• {base_name}_metrics.json (Performance metrics)\n"
                f"• {base_name}_comparison_plot.png (Comparison plot)\n\n"
                f"Directory: {output_dir}")
            
    def create_validation_comparison_plot(self, export_events, save_path=None):
        """Create a comparison plot showing original vs validated motion energy classification"""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        
        # Calculate statistics
        accepted_count = sum(1 for event in export_events if event[3] == 'accepted')
        edited_count = sum(1 for event in export_events if event[3] == 'edited')
        rejected_count = sum(1 for event in export_events if event[3] == 'rejected')
        manually_added_count = sum(1 for event in export_events if event[3] == 'manually added')
        pending_count = sum(1 for event in export_events if event[3] == 'pending')
        
        total_events = len(export_events)
        true_positives = accepted_count + edited_count  # Accepted + Edited
        false_positives = rejected_count  # Rejected
        
        # Check if motion_energy exists
        if self.motion_energy is None:
            QMessageBox.warning(self, "Warning", "No motion energy data available for comparison plot")
            return
            
        # Create the comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Top subplot: Original classification (all events as accepted)
        ax1.plot(np.arange(len(self.motion_energy)), self.motion_energy, color='blue', linewidth=1, alpha=0.7)
        
        # Plot all original events as accepted (purple for twitch, yellow for active)
        for event in export_events:
            onset, offset, event_type, status, score = event
            if event_type == 'twitch':
                ax1.axvspan(onset, offset, color='purple', alpha=0.6)
            elif event_type == 'active':
                ax1.axvspan(onset, offset, color='yellow', alpha=0.6)
        
        ax1.set_title('Original Motion Energy Classification (All Events)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Motion Energy', fontsize=12)
        ax1.set_xlim(0, len(self.motion_energy))
        ax1.set_ylim(0, max(self.motion_energy) * 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot: Validated classification (with validation markers)
        ax2.plot(np.arange(len(self.motion_energy)), self.motion_energy, color='blue', linewidth=1, alpha=0.7)
        
        # Plot events with validation status
        for event in export_events:
            onset, offset, event_type, status, score = event
            
            # Set alpha based on validation status
            if status == 'accepted':
                alpha = 1.0
            elif status == 'edited':
                alpha = 0.8
            elif status == 'rejected':
                alpha = 0.3
            elif status == 'manually added':
                alpha = 0.5
            else:  # pending
                alpha = 0.4
                
            if event_type == 'twitch':
                ax2.axvspan(onset, offset, color='purple', alpha=alpha)
            elif event_type == 'active':
                ax2.axvspan(onset, offset, color='yellow', alpha=alpha)
        
        # Add validation markers
        for event in export_events:
            onset, offset, event_type, status, score = event
            if status == 'accepted':
                ax2.plot(onset, max(self.motion_energy) * 1.05, 'o', color='green', markersize=6)
            elif status == 'rejected':
                ax2.plot(onset, max(self.motion_energy) * 1.05, 'o', color='red', markersize=6)
            elif status == 'edited':
                ax2.plot(onset, max(self.motion_energy) * 1.05, 'o', color='orange', markersize=6)
            elif status == 'manually added':
                ax2.plot(onset, max(self.motion_energy) * 1.05, 'o', color='blue', markersize=6)
        
        ax2.set_title('Validated Motion Energy Classification', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel('Motion Energy', fontsize=12)
        ax2.set_xlim(0, len(self.motion_energy))
        ax2.set_ylim(0, max(self.motion_energy) * 1.2)
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='purple', alpha=0.6, linewidth=10, label='Twitch'),
            Line2D([0], [0], color='yellow', alpha=0.6, linewidth=10, label='Active'),
            Line2D([0], [0], marker='o', color='green', markersize=8, label='Accepted', linestyle=''),
            Line2D([0], [0], marker='o', color='red', markersize=8, label='Rejected', linestyle=''),
            Line2D([0], [0], marker='o', color='orange', markersize=8, label='Edited', linestyle=''),
            Line2D([0], [0], marker='o', color='blue', markersize=8, label='Manually Added', linestyle='')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Show statistics in a text box
        stats_text = f"""
Validation Statistics:
- Total Events: {total_events}
- Accepted: {accepted_count} ({accepted_count/total_events*100:.1f}%)
- Edited: {edited_count} ({edited_count/total_events*100:.1f}%)
- Rejected: {rejected_count} ({rejected_count/total_events*100:.1f}%)
- Manually Added: {manually_added_count} ({manually_added_count/total_events*100:.1f}%)
- Pending: {pending_count} ({pending_count/total_events*100:.1f}%)

Performance Metrics:
- True Positives (Accepted + Edited): {true_positives} ({true_positives/total_events*100:.1f}%)
- False Positives (Rejected): {false_positives} ({false_positives/total_events*100:.1f}%)
- Precision: {true_positives/(true_positives+false_positives)*100:.1f}% (if no pending/manually added)
        """
        
        # Add statistics text box
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
            
    # Removed save_performance_metrics() - functionality merged into save_and_export_validation()

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

    def zoom_in_timeline(self):
        # Zoom in on the x-axis by a factor of 2, centered on current frame
        ax = self.timeline_canvas.ax
        xlim = ax.get_xlim()
        center = self.timeline_canvas.current_frame
        width = xlim[1] - xlim[0]
        new_width = width / 2
        new_xlim = (max(center - new_width/2, 0), min(center + new_width/2, self.timeline_canvas.total_frames))
        ax.set_xlim(new_xlim)
        self.timeline_canvas.draw()
    def zoom_out_timeline(self):
        # Zoom out on the x-axis by a factor of 2, centered on current frame
        ax = self.timeline_canvas.ax
        xlim = ax.get_xlim()
        center = self.timeline_canvas.current_frame
        width = xlim[1] - xlim[0]
        new_width = width * 2
        new_xlim = (max(center - new_width/2, 0), min(center + new_width/2, self.timeline_canvas.total_frames))
        ax.set_xlim(new_xlim)
        self.timeline_canvas.draw()

    def reset_zoom_timeline(self):
        # Reset the x and y axis to initial state (full view)
        ax = self.timeline_canvas.ax
        ax.set_xlim(0, max(self.timeline_canvas.total_frames, 1000))
        ax.set_ylim(0, 1.2)
        self.timeline_canvas.draw()

    def zoom_in_video(self):
        """Zoom in on the video"""
        self.video_zoom_factor = min(self.video_zoom_factor * 1.2, 3.0)  # Max 3x zoom
        if self.cap is not None:
            self.show_frame(self.current_frame)
            
    def zoom_out_video(self):
        """Zoom out on the video"""
        self.video_zoom_factor = max(self.video_zoom_factor / 1.2, 0.3)  # Min 0.3x zoom
        if self.cap is not None:
            self.show_frame(self.current_frame)
            
    def reset_video_zoom(self):
        """Reset video zoom to original size"""
        self.video_zoom_factor = 1.0
        if self.cap is not None:
            self.show_frame(self.current_frame)

    def update_onset_status(self):
        """Update the onset status bar with current frame info"""
        onset_type = self.onset_types.get(self.current_frame, None)
        if onset_type:
            self.onset_status_label.setText(f"Frame {self.current_frame}: {onset_type.upper()} ONSET")
            # Color code based on onset type
            if onset_type == 'twitch':
                self.onset_status_label.setStyleSheet("background-color: purple; color: white; padding: 5px; border: 1px solid gray; font-weight: bold;")
            elif onset_type == 'active':
                self.onset_status_label.setStyleSheet("background-color: yellow; color: black; padding: 5px; border: 1px solid gray; font-weight: bold;")
        else:
            self.onset_status_label.setText(f"Frame {self.current_frame}: No onset")
            self.onset_status_label.setStyleSheet("background-color: lightgray; padding: 5px; border: 1px solid gray;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    annotator = MotionAnnotator()
    annotator.show()
    sys.exit(app.exec_()) 
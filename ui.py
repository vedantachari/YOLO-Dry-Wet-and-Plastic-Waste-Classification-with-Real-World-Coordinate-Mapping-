import os
import json
import threading
import queue
import time
from functools import partial
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.modalview import ModalView
from kivy.uix.label import Label
from camera_calibration import calibrate_camera, draw_corners

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    print("ultralytics import failed:", e)
    ULTRALYTICS_AVAILABLE = False

# Optional ROS
try:
    import rospy
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
    ROS_AVAILABLE = True
except Exception:
    rospy = None
    ROS_AVAILABLE = False

KV = r'''
#:set primary_color (0.16, 0.71, 0.29, 1)  # Soft green
#:set text_color (1, 1, 1, 1)  # Pure white for better visibility
#:set bg_color (0.15, 0.15, 0.15, 1)  # Dark grey
#:set widget_bg (0.25, 0.25, 0.25, 1)  # Lighter grey for widgets

<RobotPositionPopup@ModalView>:
    size_hint: 0.6, 0.4
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 10
        StyledLabel:
            text: 'Robot Base Position (meters)'
            size_hint_y: None
            height: '40dp'
        BoxLayout:
            size_hint_y: None
            height: '40dp'
            spacing: 10
            StyledLabel:
                text: 'X:'
                size_hint_x: None
                width: '30dp'
            StyledTextInput:
                id: robot_x
                text: '0.0'
                input_filter: 'float'
            StyledLabel:
                text: 'Y:'
                size_hint_x: None
                width: '30dp'
            StyledTextInput:
                id: robot_y
                text: '0.0'
                input_filter: 'float'
            StyledLabel:
                text: 'Z:'
                size_hint_x: None
                width: '30dp'
            StyledTextInput:
                id: robot_z
                text: '0.0'
                input_filter: 'float'
        BoxLayout:
            size_hint_y: None
            height: '40dp'
            spacing: 10
            Widget:
                size_hint_x: 0.7
            RoundedButton:
                text: 'Set Position'
                size_hint_x: 0.3
                on_release: app.root.set_robot_position(float(robot_x.text), float(robot_y.text), float(robot_z.text))

<RoundedButton@Button>:
    background_color: 0,0,0,0
    color: text_color
    canvas.before:
        Color:
            rgba: primary_color if self.state == 'normal' else (0.14, 0.61, 0.25, 1)
        RoundedRectangle:
            size: self.size
            pos: self.pos
            radius: [5,]

<RoundedSpinner@Spinner>:
    background_color: 0,0,0,0
    color: text_color
    canvas.before:
        Color:
            rgba: widget_bg
        RoundedRectangle:
            size: self.size
            pos: self.pos
            radius: [5,]

<StyledLabel@Label>:
    color: text_color
    bold: True

<StyledTextInput@TextInput>:
    multiline: False

<CameraImage>:
    allow_stretch: True
    keep_ratio: True
    canvas.before:
        Color:
            rgba: (0.2, 0.2, 0.2, 1)  # Slightly lighter than background
        RoundedRectangle:
            size: self.size
            pos: self.pos
            radius: [10,]

<RootWidget>:
    orientation: 'vertical'
    padding: 10
    spacing: 10
    canvas.before:
        Color:
            rgba: bg_color
        Rectangle:
            size: self.size
            pos: self.pos

    BoxLayout:
        size_hint_y: None
        height: '40dp'
        spacing: 10
        RoundedButton:
            text: 'Load YOLO .pt'
            on_release: root.open_model_filechooser()
        StyledLabel:
            text: root.model_path or 'No model'
            size_hint_x: 1

    BoxLayout:
        size_hint_y: None
        height: '40dp'
        spacing: 10
        
        # Left half: buttons in a horizontal box
        BoxLayout:
            size_hint_x: 0.5
            spacing: 10
            RoundedButton:
                text: 'Load camera calib (YAML)'
                on_release: root.open_calib_filechooser()
            RoundedButton:
                text: 'Calibrate Camera'
                on_release: root.start_camera_calibration()

        
        # Right half: label
        StyledLabel:
            text: root.calib_path or 'No calib'
            size_hint_x: 0.5
            text_size: self.size
            halign: 'center'
            valign: 'middle'

    BoxLayout:
        size_hint_y: None
        height: '40dp'
        spacing: 10
        RoundedButton:
            text: 'Load homography (npy/txt)'
            on_release: root.open_homography_filechooser()
        StyledLabel:
            text: root.homography_path or 'No homography'
            size_hint_x: 1

    BoxLayout:
        size_hint_y: None
        height: '40dp'
        spacing: 10

        StyledLabel:
            text: 'Depth source:'
            size_hint_x: None
            width: '100dp'
        RoundedSpinner:
            id: depth_spinner
            text: 'None'
            values: ['None','Depth Video','RealSense (pyrealsense2)']
            size_hint_x: None
            width: '200dp'
            on_text: root.on_depth_source(self.text)

        StyledLabel:
            text: 'Plane Z (m):'
            size_hint_x: None
            width: '90dp'
        StyledTextInput:
            id: plane_z
            text: str(root.plane_z)
            input_filter: 'float'
            size_hint_x: None
            width: '80dp'
            on_text_validate: root.set_plane_z(self.text)

    BoxLayout:
        size_hint_y: None
        height: '40dp'
        spacing: 10
        StyledLabel:
            text: 'Coordinate output:'
            size_hint_x: None
            width: '140dp'
        RoundedSpinner:
            id: coord_spinner
            text: 'World (X,Y,Z)'
            values: ['Image (u,v)','Camera (x,y,z)','World (X,Y,Z)','Joint Angles (theta1..3)','ROS Pose']
            size_hint_x: None
            width: '220dp'
            on_text: root.on_coord_mode(self.text)

        RoundedButton:
            text: 'Send All'
            on_press: root.send_all_current_detections()

    BoxLayout:
        spacing: 6

        CameraImage:
            id: image_view
            size_hint_x: 0.75

        BoxLayout:
            orientation: 'vertical'
            size_hint_x: 0.25
            spacing: 6
            RoundedSpinner:
                id: input_spinner
                text: 'Camera'
                values: ['Camera', 'Video File']
                size_hint_y: None
                height: '40dp'
                on_text: root.on_input_source(self.text)
            RoundedButton:
                text: 'Start'
                on_press: root.start_processing()
            RoundedButton:
                text: 'Stop'
                on_press: root.stop_processing()
            StyledLabel:
                text: 'FPS: ' + str(root.fps)
            StyledLabel:
                id: status_label
                text: root.status
                text_size: self.size
                halign: 'left'
                valign: 'top'

    BoxLayout:
        size_hint_y: None
        height: '40dp'
        spacing: 10
        StyledLabel:
            text: 'Robot arm link lengths (m) for IK:'
            size_hint_x: None
            width: '220dp'
        StyledTextInput:
            id: l1; text: str(root.l1); input_filter:'float'; size_hint_x:None; width:'70dp'
        StyledTextInput:
            id: l2; text: str(root.l2); input_filter:'float'; size_hint_x:None; width:'70dp'
        StyledTextInput:
            id: l3; text: str(root.l3); input_filter:'float'; size_hint_x:None; width:'70dp'
        RoundedButton:
            text: 'Set Robot Position'
            size_hint_x: None
            width: '140dp'
            on_release: root.show_robot_position_popup()
'''

class CameraImage(Image):
    pass

class CalibrationPopup(ModalView):
    def __init__(self, on_calibrate=None, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.8, 0.8)
        self.auto_dismiss = False
        
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Pattern size inputs
        pattern_layout = BoxLayout(size_hint_y=None, height='40dp')
        pattern_layout.add_widget(Label(text='Pattern size (WxH):'))
        self.width_input = TextInput(
            text='9', 
            multiline=False, 
            input_filter='int', 
            size_hint_x=None, 
            width='60dp'
        )
        self.height_input = TextInput(
            text='6', 
            multiline=False, 
            input_filter='int', 
            size_hint_x=None, 
            width='60dp'
        )
        pattern_layout.add_widget(self.width_input)
        pattern_layout.add_widget(Label(text='x', color=(1,1,1,1)))
        pattern_layout.add_widget(self.height_input)
        layout.add_widget(pattern_layout)
        
        # Square size input
        square_layout = BoxLayout(size_hint_y=None, height='40dp')
        square_layout.add_widget(Label(text='Square size (meters):'))
        self.square_input = TextInput(
            text='0.025', 
            multiline=False, 
            input_filter='float', 
            size_hint_x=None, 
            width='100dp'
        )
        square_layout.add_widget(self.square_input)
        layout.add_widget(square_layout)
        
        # Buttons
        btn_layout = BoxLayout(size_hint_y=None, height='40dp', spacing=10)
        capture_btn = Button(text='Capture Frame', on_release=self.capture_frame)
        calibrate_btn = Button(text='Calibrate', on_release=self.calibrate)
        close_btn = Button(text='Close', on_release=self.dismiss)
        btn_layout.add_widget(capture_btn)
        btn_layout.add_widget(calibrate_btn)
        btn_layout.add_widget(close_btn)
        layout.add_widget(btn_layout)
        
        # Preview image
        self.preview = Image()
        layout.add_widget(self.preview)
        
        # Status label
        self.status_label = Label(text='Ready', size_hint_y=None, height='30dp')
        layout.add_widget(self.status_label)
        
        self.add_widget(layout)
        self.captured_frames = []
        self.on_calibrate = on_calibrate

    def capture_frame(self, instance):
        if not hasattr(self, 'root_widget') or not self.root_widget._capture:
            self.status_label.text = 'Camera not running'
            return
            
        ret, frame = self.root_widget._capture.read()
        if not ret:
            self.status_label.text = 'Failed to capture frame'
            return
            
        # Try to find checkerboard
        pattern_size = (int(self.width_input.text), int(self.height_input.text))
        ret, frame_with_corners = draw_corners(frame, pattern_size)
        
        if ret:
            self.captured_frames.append(frame)
            self.status_label.text = f'Captured frame {len(self.captured_frames)}'
        else:
            self.status_label.text = 'No checkerboard found'
            
        # Display the frame
        h, w = frame_with_corners.shape[:2]
        frame_rgb = cv2.cvtColor(frame_with_corners, cv2.COLOR_BGR2RGB)
        texture = Texture.create(size=(w, h), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.preview.texture = texture

    def calibrate(self, instance):
        if len(self.captured_frames) < 5:
            self.status_label.text = 'Need at least 5 frames for calibration'
            return
            
        try:
            pattern_size = (int(self.width_input.text), int(self.height_input.text))
            square_size = float(self.square_input.text)
            
            ret, mtx, dist, rvecs, tvecs = calibrate_camera(
                self.captured_frames, 
                square_size=square_size,
                pattern_size=pattern_size
            )
            
            # Save calibration
            fs = cv2.FileStorage('camera_calib.yaml', cv2.FILE_STORAGE_WRITE)
            fs.write('camera_matrix', mtx)
            fs.write('dist_coeff', dist)
            fs.write('pattern_size', pattern_size)
            fs.write('square_size', square_size)
            # Add transform to move origin to bottom center
            pattern_width = pattern_size[0] * square_size
            pattern_height = pattern_size[1] * square_size
            # Create transform matrix to move origin to bottom center
            T = np.array([
                [1, 0, 0, -pattern_width/2],  # Center X
                [0, 1, 0, pattern_height],     # Bottom Y
                [0, 0, 1, 0],                 # Keep Z same
                [0, 0, 0, 1]
            ])
            fs.write('origin_transform', T)
            fs.release()
            
            self.status_label.text = f'Calibration complete! RMS: {ret:.4f}'
            
            if self.on_calibrate:
                self.on_calibrate('camera_calib.yaml')
                
        except Exception as e:
            self.status_label.text = f'Calibration failed: {str(e)}'

class RootWidget(BoxLayout):
    model_path = StringProperty('')
    calib_path = StringProperty('')
    homography_path = StringProperty('')
    status = StringProperty('Idle')
    fps = NumericProperty(0.0)
    plane_z = NumericProperty(0.0)  # plane Z in world coords (meters)
    l1 = NumericProperty(0.3)
    l2 = NumericProperty(0.2)
    l3 = NumericProperty(0.1)
    calibration_popup = ObjectProperty(None)

    _model = None
    _class_names = None
    _frame_q = queue.Queue(maxsize=4)
    _display_q = queue.Queue(maxsize=2)
    _running = False
    _capture = None
    _depth_source = 'None'
    _depth_cap = None
    _use_realsense = False

    # camera intrinsics/extrinsics
    cam_K = None         # 3x3 camera matrix
    cam_dist = None      # distortion coeffs
    extr_R = None        # 3x3 rotation from camera->world (optional)
    extr_t = None        # 3x1 translation from camera->world (optional)
    homography = None    # 3x3 image->world plane homography (optional)

    # Robot base position in world coordinates
    robot_base_x = NumericProperty(0.0)
    robot_base_y = NumericProperty(0.0)
    robot_base_z = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_interval(self._update_image_from_queue, 1/30.)
        
    def set_robot_position(self, x, y, z):
        """Set the robot base position in world coordinates"""
        self.robot_base_x = float(x)
        self.robot_base_y = float(y)
        self.robot_base_z = float(z)
        self.status = f'Robot base position set to ({x:.3f}, {y:.3f}, {z:.3f})'
        if hasattr(self, '_robot_popup'):
            self._robot_popup.dismiss()

    def show_robot_position_popup(self):
        """Show popup to set robot base position"""
        from kivy.factory import Factory
        from kivy.uix.gridlayout import GridLayout
        from kivy.uix.label import Label
        
        # Create main popup
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Add explanation of coordinate system
        explanation = (
            "World Coordinate System Reference:\n\n"
            "If using camera calibration:\n"
            "- Origin (0,0,0) is at the bottom center of the calibration pattern\n"
            "- X: right along pattern width (positive right)\n"
            "- Y: up from bottom center (positive up)\n"
            "- Z: outward from pattern surface (positive toward camera)\n\n"
            "If using homography:\n"
            "- Origin is at your defined reference point\n"
            "- Measure robot base position relative to this point"
        )
        content.add_widget(Label(
            text=explanation,
            text_size=(400, None),
            size_hint_y=None,
            height=200,
            halign='left',
            valign='top'
        ))
        
        # Create and add the position input popup
        popup = Factory.RobotPositionPopup()
        popup.ids.robot_x.text = str(self.robot_base_x)
        popup.ids.robot_y.text = str(self.robot_base_y)
        popup.ids.robot_z.text = str(self.robot_base_z)
        
        content.add_widget(popup)
        
        # Create the complete popup
        from kivy.uix.popup import Popup
        main_popup = Popup(
            title='Set Robot Base Position',
            content=content,
            size_hint=(0.8, 0.8)
        )
        self._robot_popup = main_popup
        main_popup.open()

    # ---------------- file choosers ----------------
    def open_model_filechooser(self):
        from kivy.uix.filechooser import FileChooserListView
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        chooser = FileChooserListView(filters=['*.pt'], path=os.getcwd())
        content = BoxLayout(orientation='vertical')
        content.add_widget(chooser)
        btns = BoxLayout(size_hint_y=None, height='40dp')
        ok = Button(text='Load'); cancel = Button(text='Cancel')
        btns.add_widget(ok); btns.add_widget(cancel)
        content.add_widget(btns)
        popup = Popup(title='Select YOLO model', content=content, size_hint=(0.9,0.9))
        def on_ok(instance):
            sel = chooser.selection
            if sel:
                self.model_path = sel[0]
                self.status = f'Loading model...'
                threading.Thread(target=self._load_model, args=(self.model_path,), daemon=True).start()
            popup.dismiss()
        ok.bind(on_release=on_ok); cancel.bind(on_release=lambda x: popup.dismiss())
        popup.open()

    def open_calib_filechooser(self):
        from kivy.uix.filechooser import FileChooserListView
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        chooser = FileChooserListView(filters=['*.yml','*.yaml','*.json'], path=os.getcwd())
        content = BoxLayout(orientation='vertical'); content.add_widget(chooser)
        btns = BoxLayout(size_hint_y=None, height='40dp')
        ok = Button(text='Load'); cancel = Button(text='Cancel')
        btns.add_widget(ok); btns.add_widget(cancel); content.add_widget(btns)
        popup = Popup(title='Select camera calibration', content=content, size_hint=(0.9,0.9))
        def on_ok(instance):
            sel = chooser.selection
            if sel:
                self.calib_path = sel[0]
                try:
                    self._load_camera_calibration(self.calib_path)
                    self.status = f'Calib loaded'
                except Exception as e:
                    self.status = f'Calib load failed: {e}'
            popup.dismiss()
        ok.bind(on_release=on_ok); cancel.bind(on_release=lambda x: popup.dismiss())
        popup.open()

    def open_homography_filechooser(self):
        from kivy.uix.filechooser import FileChooserListView
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.button import Button
        chooser = FileChooserListView(filters=['*.npy','*.txt','*.csv'], path=os.getcwd())
        content = BoxLayout(orientation='vertical'); content.add_widget(chooser)
        btns = BoxLayout(size_hint_y=None, height='40dp')
        ok = Button(text='Load'); cancel = Button(text='Cancel')
        btns.add_widget(ok); btns.add_widget(cancel); content.add_widget(btns)
        popup = Popup(title='Select homography', content=content, size_hint=(0.9,0.9))
        def on_ok(instance):
            sel = chooser.selection
            if sel:
                p = sel[0]
                try:
                    if p.lower().endswith('.npy'):
                        H = np.load(p)
                    else:
                        H = np.loadtxt(p)
                    H = np.asarray(H).reshape(3,3)
                    self.homography = H
                    self.homography_path = p
                    self.status = 'Homography loaded'
                except Exception as e:
                    self.status = f'Homography load failed: {e}'
            popup.dismiss()
        ok.bind(on_release=on_ok); cancel.bind(on_release=lambda x: popup.dismiss())
        popup.open()

    def _load_model(self, path):
        try:
            if not ULTRALYTICS_AVAILABLE:
                raise RuntimeError("ultralytics not installed")
            model = YOLO(path)
            self._model = model
            self._class_names = getattr(model, 'names', None)
            # If class names are missing or not a list, set manually
            if not isinstance(self._class_names, (list, tuple)):
                self._class_names = ["Dry Waste", "Wet Waste", "Plastic"]
            self.status = f'Model loaded: {os.path.basename(path)}'
        except Exception as e:
            self._model = None
            self.status = f'Model load failed: {e}'

    # ---------- calibration/homography/depth helpers ----------
    def start_camera_calibration(self):
        if not self._capture or not self._running:
            self.status = 'Start camera first'
            return
            
        if self.calibration_popup is None:
            self.calibration_popup = CalibrationPopup(on_calibrate=self.on_calibration_complete)
            self.calibration_popup.root_widget = self
            
        self.calibration_popup.captured_frames = []
        self.calibration_popup.status_label.text = 'Ready'
        self.calibration_popup.open()
        
    def on_calibration_complete(self, calib_path):
        """Called when calibration is complete with path to calibration file"""
        self.calib_path = calib_path
        try:
            self._load_camera_calibration(calib_path)
            self.status = 'Camera calibration loaded'
        except Exception as e:
            self.status = f'Error loading calibration: {e}'

    def _load_camera_calibration(self, path):
        # Accept YAML or JSON containing camera_matrix, dist_coeffs, optionally extrinsics R,t (camera->world)
        if path.lower().endswith(('.yml','.yaml')):
            fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise RuntimeError("Cannot open YAML")
            camK = fs.getNode("camera_matrix").mat()
            dist = fs.getNode("dist_coeff").mat()
            pattern_size = fs.getNode("pattern_size").mat()  # Get checkerboard size
            square_size = float(fs.getNode("square_size").real())
            # optionally extrinsics
            R = fs.getNode("R").mat()
            t = fs.getNode("t").mat()
            fs.release()
            if camK is None:
                raise RuntimeError("camera_matrix missing")
            self.cam_K = np.array(camK, dtype=float)
            self.cam_dist = None if dist is None else np.array(dist, dtype=float).reshape(-1,1)
            # Load pattern info for coordinate transform
            pattern_size = fs.getNode("pattern_size").mat()
            square_size = float(fs.getNode("square_size").real())
            T = fs.getNode("origin_transform").mat()

            if R is not None and t is not None and R.size==9:
                # Convert original transform to 4x4 matrix
                transform = np.eye(4)
                transform[:3,:3] = np.array(R).reshape(3,3)
                transform[:3,3] = np.array(t).reshape(3)
                
                # Apply origin shift to bottom center
                transform = T @ transform
                
                # Extract new R and t
                self.extr_R = transform[:3,:3]
                self.extr_t = transform[:3,3].reshape(3,1)
            else:
                self.extr_R = None
                self.extr_t = None
                
            # Store pattern info for reference
            self.pattern_width = pattern_size[0] * square_size
            self.pattern_height = pattern_size[1] * square_size
        else:
            # try JSON
            with open(path,'r') as fh:
                data = json.load(fh)
            self.cam_K = np.array(data['camera_matrix'], dtype=float)
            self.cam_dist = np.array(data.get('dist_coeff', []), dtype=float).reshape(-1,1) if data.get('dist_coeff') else None
            if 'R' in data and 't' in data:
                self.extr_R = np.array(data['R'], dtype=float).reshape(3,3)
                self.extr_t = np.array(data['t'], dtype=float).reshape(3,1)
            else:
                self.extr_R = None
                self.extr_t = None
    
    _input_source = 'Camera'
    _video_path = None

    def on_input_source(self, text):
        self._input_source = text
        if text == 'Video File':
            # open file chooser for video file
            from kivy.uix.filechooser import FileChooserListView
            from kivy.uix.popup import Popup
            from kivy.uix.boxlayout import BoxLayout
            from kivy.uix.button import Button
            chooser = FileChooserListView(filters=['*.mp4','*.avi','*.mov','*.mkv'], path=os.getcwd())
            content = BoxLayout(orientation='vertical'); content.add_widget(chooser)
            btns = BoxLayout(size_hint_y=None, height='40dp'); ok=Button(text='Select'); cancel=Button(text='Cancel')
            btns.add_widget(ok); btns.add_widget(cancel); content.add_widget(btns)
            popup = Popup(title='Select video file', content=content, size_hint=(0.9,0.9))
            def on_ok(instance):
                sel = chooser.selection
                if sel:
                    self._video_path = sel[0]
                    self.status = f'Video selected: {os.path.basename(self._video_path)}'
                popup.dismiss()
            ok.bind(on_release=on_ok); cancel.bind(on_release=lambda x: popup.dismiss())
            popup.open()
        else:
            self._video_path = None
            self.status = 'Camera selected'

    def on_depth_source(self, text):
        self._depth_source = text
        if text == 'Depth Video':
            # open file chooser for depth video
            from kivy.uix.filechooser import FileChooserListView
            from kivy.uix.popup import Popup
            from kivy.uix.boxlayout import BoxLayout
            from kivy.uix.button import Button
            chooser = FileChooserListView(filters=['*.npy','*.png','*.mp4','*.avi'], path=os.getcwd())
            content = BoxLayout(orientation='vertical'); content.add_widget(chooser)
            btns = BoxLayout(size_hint_y=None, height='40dp'); ok=Button(text='Load'); cancel=Button(text='Cancel')
            btns.add_widget(ok); btns.add_widget(cancel); content.add_widget(btns)
            popup = Popup(title='Select depth source', content=content, size_hint=(0.9,0.9))
            def on_ok(instance):
                sel = chooser.selection
                if sel:
                    p = sel[0]
                    self._depth_path = p
                    # try opening as video first
                    try:
                        cap = cv2.VideoCapture(p)
                        if cap.isOpened():
                            self._depth_cap = cap
                            self.status = 'Depth video loaded'
                        else:
                            # try npy stack
                            arr = np.load(p)
                            self._depth_cap = arr  # numpy array of depth frames
                            self.status = 'Depth npy loaded'
                    except Exception as e:
                        self.status = f'Depth load failed: {e}'
                popup.dismiss()
            ok.bind(on_release=on_ok); cancel.bind(on_release=lambda x: popup.dismiss())
            popup.open()
        elif text == 'RealSense (pyrealsense2)':
            try:
                import pyrealsense2 as rs
                self._use_realsense = True
                self.status = 'RealSense selected (ensure device connected)'
                # Note: full RealSense handling will be started when processing starts
            except Exception as e:
                self._use_realsense = False
                self.status = f'RealSense not available: {e}'
        else:
            self._depth_cap = None
            self._use_realsense = False
            self._depth_path = None
            self.status = 'No depth'

    def set_plane_z(self, text):
        try:
            self.plane_z = float(text)
            self.status = f'Plane Z set to {self.plane_z} m'
        except:
            pass

    def on_coord_mode(self, mode):
        """Called when coordinate output mode is changed in spinner."""
        if not hasattr(self, '_latest_detections'):
            return
        # Update status with current mode and detection count
        count = len(self._latest_detections) if self._latest_detections else 0
        self.status = f'Coordinate mode: {mode} ({count} detections)'

    # ---------------- start/stop capture & inference ----------------
    def start_processing(self):
        if self._running:
            self.status = 'Already running'
            return
        if self._model is None:
            self.status = 'Load a YOLO model first'
            return
        # Select input source
        if self._input_source == 'Camera':
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.status = 'Cannot open camera'
                return
            self._capture = cap
            self.status = 'Camera started'
        elif self._input_source == 'Video File':
            if not self._video_path:
                self.status = 'Select a video file first'
                return
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.status = f'Cannot open video: {self._video_path}'
                return
            self._capture = cap
            self.status = f'Video started: {os.path.basename(self._video_path)}'
        else:
            self.status = 'Unknown input source'
            return
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()

    def stop_processing(self):
        self._running = False
        try:
            if self._capture:
                self._capture.release()
        except: pass
        if self._depth_cap and hasattr(self._depth_cap, 'release'):
            try: self._depth_cap.release()
            except: pass
        self.status = 'Stopped'

    def _capture_loop(self):
        cap = self._capture
        consecutive_failures = 0
        while self._running and cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        print("Multiple frame capture failures - camera may be disconnected")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                
                # Ensure frame is in correct format
                if frame.dtype != np.uint8:
                    frame = cv2.convertScaleAbs(frame)
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # optionally read depth frame synchronized
                depth_frame = None
                if self._depth_cap is not None:
                    try:
                        if isinstance(self._depth_cap, cv2.VideoCapture):
                            ok, dfr = self._depth_cap.read()
                            if ok and dfr is not None:
                                # if depth video encoded as 16-bit PNG-like values:
                                depth_frame = dfr.astype(np.float32) / 1000.0  # if mm stored: convert to meters
                        elif isinstance(self._depth_cap, np.ndarray):
                            # assume numpy stack: pick frame by modulo
                            idx = int(time.time()*30) % len(self._depth_cap)
                            depth_frame = self._depth_cap[idx].astype(np.float32)
                    except Exception as e:
                        print(f"Depth capture error: {e}")
                
                # store tuple (rgb, depth) - clear queue if full
                if self._frame_q.full():
                    try:
                        while not self._frame_q.empty():
                            _ = self._frame_q.get_nowait()
                    except queue.Empty:
                        pass
                
                self._frame_q.put((frame, depth_frame), timeout=0.05)
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
            
            # Small delay to prevent busy-waiting
            time.sleep(0.001)

    def _inference_loop(self):
        t0 = time.time(); count = 0
        while self._running:
            # Get frame with timeout
            try:
                frame, depth = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error getting frame from queue: {e}")
                continue

            if frame is None:
                continue

            try:
                # Make a safe copy
                annotated = frame.copy()
                
                # Run inference with error handling
                try:
                    results = self._model(frame, conf=0.25)
                    res0 = results[0] if isinstance(results, (list,tuple)) else results
                    annotated = self._draw_and_compute(frame, depth, annotated, res0)
                except Exception as e:
                    error_msg = str(e)[:50]  # Truncate very long error messages
                    cv2.putText(annotated, f'Inference err: {error_msg}', 
                              (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print(f"Inference error: {e}")

                # Update display queue - clear if full
                if self._display_q.full():
                    try:
                        while not self._display_q.empty():
                            _ = self._display_q.get_nowait()
                    except queue.Empty:
                        pass

                self._display_q.put(annotated)

                # Update FPS calculation
                count += 1
                if count >= 5:
                    dt = time.time() - t0
                    fps_now = count/dt if dt>0 else 0.0
                    self.fps = round((self.fps*0.8 + fps_now*0.2), 2)
                    t0 = time.time(); count=0

            except Exception as e:
                print(f"Error in inference loop: {e}")
                time.sleep(0.1)  # Prevent rapid error loops

    # ---------------- mapping / coordinate computations ----------------
    def _pixel_to_camera_xyz(self, u, v, depth_m):
        """Given pixel coords u,v and depth in meters, return camera coords (x,y,z) in meters."""
        if self.cam_K is None:
            raise RuntimeError("Camera intrinsics not loaded")
        fx = self.cam_K[0,0]; fy = self.cam_K[1,1]
        cx = self.cam_K[0,2]; cy = self.cam_K[1,2]
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        return np.array([x,y,z], dtype=float).reshape(3,1)

    def _camera_to_world(self, cam_xyz):
        """If extrinsics provided (camera->world), apply transform. Else return cam_xyz."""
        if self.extr_R is None or self.extr_t is None:
            return cam_xyz.flatten()
        w = self.extr_R.dot(cam_xyz) + self.extr_t
        return w.flatten()

    def _image_to_world_plane(self, u, v):
        """Map image pixel (u,v) to world X,Y using homography. Z = plane_z."""
        if self.homography is None:
            raise RuntimeError("No homography loaded")
        uv1 = np.array([u,v,1.0], dtype=float)
        XY1 = self.homography.dot(uv1)
        XY1 = XY1 / XY1[2]
        X = float(XY1[0]); Y = float(XY1[1]); Z = float(self.plane_z)
        return np.array([X,Y,Z], dtype=float)

    # ---------------- simple 3-DOF planar IK example ----------------
    def ik_3dof_planar(self, target_xyz, phi=0.0):
        """Simple inverse kinematics for planar 3-link arm.
           target_xyz: [X,Y,Z] in world coordinates.
           Returns angles in radians (theta1,theta2,theta3) or None if unreachable.
        """
        # Convert target from world coordinates to robot base coordinates
        X = float(target_xyz[0]) - self.robot_base_x
        Y = float(target_xyz[1]) - self.robot_base_y
        Z = float(target_xyz[2]) - self.robot_base_z  # Z offset might be important
        l1 = float(self.l1); l2 = float(self.l2); l3 = float(self.l3)
        # Effective target for first 2 links: want wrist point (subtract l3 along desired phi)
        wx = X - l3 * np.cos(phi)
        wy = Y - l3 * np.sin(phi)
        D = (wx**2 + wy**2 - l1**2 - l2**2) / (2*l1*l2)
        if abs(D) > 1.0:
            return None
        theta2 = np.arctan2(np.sqrt(max(0,1-D**2)), D)
        theta1 = np.arctan2(wy, wx) - np.arctan2(l2*np.sin(theta2), l1 + l2*np.cos(theta2))
        theta3 = phi - theta1 - theta2
        return (theta1, theta2, theta3)

    # ---------------- draw / compute per-frame detection coordinates ----------------
    def _draw_and_compute(self, rgb_frame, depth_frame, annotated, res):
        # Draw coordinate system reference if we have calibration
        if self.cam_K is not None and self.extr_R is not None:
            # Draw coordinate axes at origin (0,0,0)
            axes_length = 0.1  # 10cm axes
            origin = np.array([0.0, 0.0, 0.0])
            x_axis = np.array([axes_length, 0.0, 0.0])
            y_axis = np.array([0.0, axes_length, 0.0])
            z_axis = np.array([0.0, 0.0, axes_length])
            
            # Project points to image
            points_3d = np.array([origin, x_axis, y_axis, z_axis])
            points_2d, _ = cv2.projectPoints(
                points_3d,
                cv2.Rodrigues(self.extr_R)[0],
                self.extr_t,
                self.cam_K,
                self.cam_dist
            )
            
            # Draw axes
            origin = tuple(map(int, points_2d[0].ravel()))
            x_point = tuple(map(int, points_2d[1].ravel()))
            y_point = tuple(map(int, points_2d[2].ravel()))
            z_point = tuple(map(int, points_2d[3].ravel()))
            
            cv2.line(annotated, origin, x_point, (0,0,255), 2)  # X axis in red
            cv2.line(annotated, origin, y_point, (0,255,0), 2)  # Y axis in green
            cv2.line(annotated, origin, z_point, (255,0,0), 2)  # Z axis in blue
            
            # Label axes
            cv2.putText(annotated, 'X', x_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.putText(annotated, 'Y', y_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.putText(annotated, 'Z', z_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
            # Draw robot base position if set
            if any([self.robot_base_x, self.robot_base_y, self.robot_base_z]):
                robot_pos = np.array([[self.robot_base_x, self.robot_base_y, self.robot_base_z]])
                robot_2d, _ = cv2.projectPoints(
                    robot_pos,
                    cv2.Rodrigues(self.extr_R)[0],
                    self.extr_t,
                    self.cam_K,
                    self.cam_dist
                )
                robot_point = tuple(map(int, robot_2d[0].ravel()))
                cv2.circle(annotated, robot_point, 5, (255,255,0), -1)  # Yellow dot
                cv2.putText(annotated, 'Robot Base', robot_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        boxes = []
        scores = []
        class_ids = []
        try:
            boxes_attr = getattr(res, 'boxes', None)
            if boxes_attr is not None:
                xyxy = boxes_attr.xyxy.cpu().numpy() if hasattr(boxes_attr.xyxy, 'cpu') else boxes_attr.xyxy.numpy()
                confs = boxes_attr.conf.cpu().numpy() if hasattr(boxes_attr.conf, 'cpu') else boxes_attr.conf.numpy()
                cls_idx = boxes_attr.cls.cpu().numpy().astype(int) if hasattr(boxes_attr.cls, 'cpu') else boxes_attr.cls.numpy().astype(int)
                boxes = xyxy
                scores = confs
                class_ids = cls_idx

                # Set default camera parameters if none provided
                if self.cam_K is None:
                    h, w = rgb_frame.shape[:2]
                    self.cam_K = np.array([
                        [w, 0, w/2],
                        [0, h, h/2],
                        [0, 0, 1]
                    ], dtype=float)
                if self.plane_z == 0:
                    self.plane_z = 1.0  # Default 1 meter depth
        except Exception:
            pass

        h,w = rgb_frame.shape[:2]
        self._latest_detections = [] 
        
        class_colors = {
            0: (0, 255, 0),
            1: (0, 0, 255),
            2: (255, 0, 128),
            3: (255, 0, 0)   
        }
        
        for i,box in enumerate(boxes if len(boxes)>0 else []):
            x1,y1,x2,y2 = [int(v) for v in box[:4]]
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
            conf = float(scores[i]) if i < len(scores) else 0.0
            class_id = int(class_ids[i])
            
            if isinstance(self._class_names, (list,tuple)) and class_id < len(self._class_names):
                cls_name = self._class_names[class_id]
            else:
                cls_name = str(class_id)
                
            # Get color for this class
            color = class_colors.get(class_id, (255, 255, 255))  # White as default
            if isinstance(self._class_names, (list,tuple)):
                try:
                    cls_name = self._class_names[class_ids[i]]
                except: pass

            # compute world / camera coordinates using simplified approach
            coord_camera = None
            coord_world = None
            used_method = None

            # Use default plane projection
            try:
                # Use center point of detection box
                depth_m = float(self.plane_z)  # Fixed depth
                cam_xyz = self._pixel_to_camera_xyz(cx, cy, depth_m)
                coord_camera = cam_xyz.flatten()
                coord_world = coord_camera  # Simple mapping: camera coords = world coords
                used_method = 'simple-projection'
            except Exception:
                depth_m = float(np.nan)
                if np.isfinite(depth_m) and depth_m>0.001:
                    try:
                        cam_xyz = self._pixel_to_camera_xyz(cx, cy, depth_m)
                        coord_camera = cam_xyz.flatten()
                        coord_world = self._camera_to_world(cam_xyz)
                        used_method = 'depth+intrinsics'
                    except Exception as e:
                        used_method = None

            # 2) fallback: homography (image->world plane)
            if coord_world is None and self.homography is not None:
                try:
                    p = self._image_to_world_plane(cx, cy)
                    coord_world = p
                    # if we have intrinsics, we can compute camera coords by inverting extrinsic if available
                    if self.extr_R is not None and self.extr_t is not None:
                        # world = R*cam + t => cam = R^T*(world - t)
                        cam = np.linalg.inv(self.extr_R).dot(p.reshape(3,1) - self.extr_t).flatten()
                        coord_camera = cam
                    used_method = 'homography'
                except Exception:
                    pass

            # 3) last resort: project ray from pixel using intrinsics and assume plane_z intersection
            if coord_world is None and self.cam_K is not None:
                try:
                    # form ray in camera frame direction
                    fx, fy = self.cam_K[0,0], self.cam_K[1,1]
                    cx0, cy0 = self.cam_K[0,2], self.cam_K[1,2]
                    x_dir = (cx - cx0)/fx
                    y_dir = (cy - cy0)/fy
                    # parametric ray: cam_p = s * [x_dir, y_dir, 1]
                    # find s such that world_z (after extrinsic) == plane_z
                    if self.extr_R is not None and self.extr_t is not None:
                        R = self.extr_R; t = self.extr_t.flatten()
                        # cam point = s * d ; world = R*(s*d) + t ; we want world_z = plane_z
                        d = np.array([x_dir, y_dir, 1.0], dtype=float)
                        A = (R[2,:].dot(d))
                        if abs(A) > 1e-6:
                            s = (self.plane_z - t[2]) / A
                            cam = s * d
                            coord_camera = cam.flatten()
                            coord_world = (R.dot(cam.reshape(3,1)) + t.reshape(3,1)).flatten()
                            used_method = 'ray-plane'
                    else:
                        # if no extrinsic, just intersect camera ray with Z = plane_z in camera coords
                        if 1.0 != 0:
                            s = self.plane_z / 1.0
                            cam = s * np.array([x_dir, y_dir, 1.0])
                            coord_camera = cam.flatten()
                            coord_world = cam.flatten()
                            used_method = 'cam-ray-plane'
                except Exception:
                    pass

            # Compose label & draw
            # Get current coordinate mode
            coord_mode = self.ids['coord_spinner'].text
            label = f'{cls_name} {conf:.2f} |'
            
            if coord_mode == 'World (X,Y,Z)' and coord_world is not None:
                label += f' XYZ: {coord_world[0]:.3f},{coord_world[1]:.3f},{coord_world[2]:.3f}'
            elif coord_mode == 'Camera (x,y,z)' and coord_camera is not None:
                label += f' xyz: {coord_camera[0]:.3f},{coord_camera[1]:.3f},{coord_camera[2]:.3f}'
            elif coord_mode == 'Image (u,v)':
                label += f' uv: {cx},{cy}'
            elif coord_mode == 'Joint Angles (theta1..3)' and coord_world is not None:
                thetas = self.ik_3dof_planar(coord_world)
                if thetas is not None:
                    label += f' θ: {thetas[0]:.3f},{thetas[1]:.3f},{thetas[2]:.3f}'
                else:
                    label += ' unreachable'
            elif coord_mode == 'ROS Pose' and coord_world is not None:
                label += f' pose: {coord_world[0]:.3f},{coord_world[1]:.3f},{coord_world[2]:.3f}'
            else:
                label += ' no data'

            # Draw bounding box with class-specific color
            cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 3)  # Made line thicker
            
            # Draw label background and text with larger font
            font_scale = 0.7  # Increased font scale
            thickness = 2     # Increased text thickness
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # Darken the class color for the label background
            dark_color = tuple(int(c * 0.7) for c in color)
            # Added more padding around text
            padding = 8
            cv2.rectangle(annotated, (x1, y1 - int(1.3*th)-padding), (x1 + tw + padding*2, y1), dark_color, -1)
            cv2.putText(annotated, label, (x1+padding, y1-padding), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
            
            # Draw center point
            cv2.circle(annotated, (cx, cy), 3, color, -1)

            # store detection info for Send All / ROS
            det_info = {
                'bbox': [int(x1),int(y1),int(x2),int(y2)],
                'center_px': [int(cx),int(cy)],
                'score': float(conf),
                'class': str(cls_name),
                'method': used_method,
                'camera_xyz': None if coord_camera is None else [float(x) for x in coord_camera],
                'world_xyz': None if coord_world is None else [float(x) for x in coord_world]
            }
            # compute joint angles if possible
            if det_info['world_xyz'] is not None:
                ik = self.ik_3dof_planar(det_info['world_xyz'])
                det_info['ik_theta'] = None if ik is None else [float(x) for x in ik]
            else:
                det_info['ik_theta'] = None
            self._latest_detections.append(det_info)
        return annotated

    # ---------------- send/publish functions ----------------
    def _format_for_output(self, det, mode):
        if mode == 'Image (u,v)':
            return {'u': det['center_px'][0], 'v': det['center_px'][1]}
        elif mode == 'Camera (x,y,z)':
            return {'camera_xyz': det['camera_xyz']}
        elif mode == 'World (X,Y,Z)':
            return {'world_xyz': det['world_xyz']}
        elif mode == 'Joint Angles (theta1..3)':
            return {'thetas': det.get('ik_theta')}
        elif mode == 'ROS Pose':
            return {'pose': det.get('world_xyz')}
        else:
            return {}

    def send_all_current_detections(self):
        # get spinner selection
        mode = self.ids['coord_spinner'].text
        if not hasattr(self, '_latest_detections') or len(self._latest_detections)==0:
            self.status = 'No detections to send'
            return
        outs = []
        for d in self._latest_detections:
            outs.append(self._format_for_output(d, mode))
        if mode == 'ROS Pose':
            if not ROS_AVAILABLE:
                self.status = 'rospy not available; saving to poses_out.json'
                with open('poses_out.json','w') as fh:
                    json.dump(outs, fh, indent=2)
                return
            # else publish
            threading.Thread(target=self._ros_publish_many, args=(outs,), daemon=True).start()
            self.status = f'Published {len(outs)} poses to ROS'
        else:
            # save to JSON and show status
            with open('coords_out.json','w') as fh:
                json.dump(outs, fh, indent=2)
            self.status = f'Saved {len(outs)} items to coords_out.json'

    def _ros_publish_many(self, outs, topic='/detected_poses', frame_id='world'):
        if not ROS_AVAILABLE:
            return
        try:
            # init node only once
            try:
                rospy.get_name()
            except:
                rospy.init_node('kivy_detection_publisher', anonymous=True)
            pub = rospy.Publisher(topic, PoseStamped, queue_size=10)
            for o in outs:
                p = o.get('pose')
                if p is None:
                    continue
                ps = PoseStamped()
                ps.header.stamp = rospy.Time.now()
                ps.header.frame_id = frame_id
                ps.pose.position.x = float(p[0]); ps.pose.position.y = float(p[1]); ps.pose.position.z = float(p[2])
                ps.pose.orientation = Quaternion(0,0,0,1)
                pub.publish(ps)
                rospy.sleep(0.05)
        except Exception as e:
            print("ROS publish error:", e)

    # ---------------- UI update ----------------
    def _update_image_from_queue(self, dt):
        try:
            frame = self._display_q.get_nowait()
        except queue.Empty:
            return

        if frame is None:
            print("Warning: Received None frame")
            return

        try:
            # Ensure frame is 3-channel uint8
            if frame.dtype != np.uint8:
                frame = cv2.convertScaleAbs(frame)
            if len(frame.shape) == 2:  # grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Frame conversion error: {e}")
            return

        try:
            h, w = img.shape[:2]
            # Ensure array is contiguous
            img = np.ascontiguousarray(img)
            
            # Create texture with explicit format
            texture = Texture.create(size=(w, h), colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            
            # Update buffer
            texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            
            # Update widget and its size
            self.ids.image_view.texture = texture
            self.ids.image_view.texture_size = (w, h)
        except Exception as e:
            print(f"Texture creation error: {e}")

class YOLOKivyCoordsApp(App):
    def build(self):
        Builder.load_string(KV)
        return RootWidget()

if __name__ == '__main__':
    YOLOKivyCoordsApp().run()
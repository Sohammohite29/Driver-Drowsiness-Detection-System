from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    Response,
    jsonify
)

import cv2
import time
import threading
import math
import winsound
import mediapipe as mp
import os
import json

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = "driver_drowsiness_secret_key"


# ============================================================
# FILES
# ============================================================

USER_FILE = "user_account.json"
DRIVER_FILE = "driver_profile.json"


# ============================================================
# USER ACCOUNT
# ============================================================

def load_user():

    if not os.path.exists(USER_FILE):
        return None

    try:
        with open(USER_FILE, "r") as file:
            return json.load(file)

    except Exception:
        return None


def save_user(username, password):

    data = {
        "username": username,
        "password": generate_password_hash(password)
    }

    with open(USER_FILE, "w") as file:
        json.dump(data, file)


def verify_user(username, password):

    user = load_user()

    if not user:
        return False

    if username.lower() != user["username"].lower():
        return False

    return check_password_hash(
        user["password"],
        password
    )


# ============================================================
# DRIVER PROFILE
# ============================================================

def load_driver():

    if not os.path.exists(DRIVER_FILE):
        return None

    try:
        with open(DRIVER_FILE, "r") as file:
            return json.load(file)

    except Exception:
        return None


def save_driver(data):

    with open(DRIVER_FILE, "w") as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ============================================================
# CAMERA
# ============================================================

camera = None


def get_camera():

    global camera

    if camera is None or not camera.isOpened():

        camera = cv2.VideoCapture(0)

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

    return camera


def release_camera():

    global camera

    if camera:

        camera.release()
        camera = None


# ============================================================
# SETTINGS
# ============================================================

EAR_THRESHOLD = 0.22

DROWSINESS_TIME = 2.0

ALARM_COOLDOWN = 3.0


# ============================================================
# MONITORING
# ============================================================

monitoring_active = True

monitoring_start_time = time.time()

total_monitoring_seconds = 0

monitoring_lock = threading.Lock()

latest_ear = 0.0

current_status = "WAITING FOR CAMERA"


# ============================================================
# SESSION DATA
# ============================================================

session_start = time.time()

session_ear_total = 0

session_ear_samples = 0

average_ear = 0

longest_drowsiness = 0


# ============================================================
# ALERT DATA
# ============================================================

alert_count = 0

alert_history = []

alert_lock = threading.Lock()


# ============================================================
# MEDIAPIPE
# ============================================================

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ============================================================
# EAR CALCULATION
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        +
        (p1[1] - p2[1]) ** 2
    )


def calculate_ear(
    landmarks,
    eye_points,
    width,
    height
):

    points = []

    for index in eye_points:

        point = landmarks[index]

        points.append(
            (
                int(point.x * width),
                int(point.y * height)
            )
        )

    p1, p2, p3, p4, p5, p6 = points

    vertical1 = distance(
        p2,
        p6
    )

    vertical2 = distance(
        p3,
        p5
    )

    horizontal = distance(
        p1,
        p4
    )

    if horizontal == 0:
        return 0

    return (
        vertical1 + vertical2
    ) / (
        2 * horizontal
    )


# ============================================================
# DRAW EYE LANDMARKS
# ============================================================

def draw_eye(
    frame,
    landmarks,
    points,
    width,
    height
):

    for index in points:

        point = landmarks[index]

        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(
            frame,
            (x, y),
            2,
            (0, 255, 0),
            -1
        )


# ============================================================
# ALARM
# ============================================================

def play_alarm():

    try:

        winsound.Beep(
            1500,
            800
        )

        time.sleep(0.2)

        winsound.Beep(
            1500,
            800
        )

    except Exception:
        pass


# ============================================================
# SAFETY SCORE
# ============================================================

def calculate_safety_score():

    score = 100 - (
        alert_count * 5
    )

    if score < 0:
        score = 0

    return score


# ============================================================
# RECORD ALERT
# ============================================================

def record_alert(
    ear,
    duration
):

    global alert_count
    global longest_drowsiness

    with alert_lock:

        alert_count += 1

        alert = {
            "number": alert_count,

            "time": time.strftime(
                "%d-%m-%Y %H:%M:%S"
            ),

            "ear": round(
                ear,
                3
            ),

            "duration": round(
                duration,
                1
            ),

            "type": "Drowsiness"
        }

        alert_history.insert(
            0,
            alert
        )

        alert_history[:] = (
            alert_history[:10]
        )

    if duration > longest_drowsiness:

        longest_drowsiness = duration


# ============================================================
# START NEW SESSION
# ============================================================

def start_session():

    global session_start
    global session_ear_total
    global session_ear_samples
    global average_ear
    global longest_drowsiness
    global alert_count
    global alert_history

    session_start = time.time()

    session_ear_total = 0

    session_ear_samples = 0

    average_ear = 0

    longest_drowsiness = 0

    with alert_lock:

        alert_count = 0

        alert_history = []


# ============================================================
# VIDEO GENERATOR
# ============================================================

def generate_frames():

    global latest_ear
    global current_status
    global session_ear_total
    global session_ear_samples
    global average_ear
    global longest_drowsiness

    closed_start = None

    last_alarm = 0

    closed_frames = 0

    open_frames = 0

    # Prevents one continuous closed-eye period
    # from being counted as multiple alerts.
    alert_recorded_for_episode = False

    while True:

        with monitoring_lock:

            active = monitoring_active

        # ----------------------------------------------------
        # MONITORING STOPPED
        # ----------------------------------------------------

        if not active:

            import numpy as np

            frame = np.zeros(
                (480, 640, 3),
                dtype=np.uint8
            )

            cv2.putText(
                frame,
                "MONITORING STOPPED",
                (150, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            ret, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if ret:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

            time.sleep(0.2)

            continue

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        cap = get_camera()

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = face_mesh.process(rgb)

        status = "NO FACE DETECTED"

        # ----------------------------------------------------
        # FACE DETECTED
        # ----------------------------------------------------

        if result.multi_face_landmarks:

            landmarks = (
                result
                .multi_face_landmarks[0]
                .landmark
            )

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE,
                width,
                height
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE,
                width,
                height
            )

            ear = (
                left_ear +
                right_ear
            ) / 2

            latest_ear = ear

            session_ear_total += ear

            session_ear_samples += 1

            average_ear = (
                session_ear_total
                /
                session_ear_samples
            )

            draw_eye(
                frame,
                landmarks,
                LEFT_EYE,
                width,
                height
            )

            draw_eye(
                frame,
                landmarks,
                RIGHT_EYE,
                width,
                height
            )

            # ------------------------------------------------
            # EYES CLOSED
            # ------------------------------------------------

            if ear < EAR_THRESHOLD:

                closed_frames += 1

                open_frames = 0

                status = "EYES CLOSED"

                if closed_frames >= 3:

                    if closed_start is None:

                        closed_start = time.time()

                    closed_time = (
                        time.time()
                        -
                        closed_start
                    )

                    cv2.putText(
                        frame,
                        f"Closed: {closed_time:.1f}s",
                        (25, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2
                    )

                    # ----------------------------------------
                    # DROWSINESS DETECTED
                    # ----------------------------------------

                    if closed_time >= DROWSINESS_TIME:

                        status = "DROWSINESS ALERT"

                        # Record only once for this
                        # continuous eye-closed episode.
                        if not alert_recorded_for_episode:

                            record_alert(
                                ear,
                                closed_time
                            )

                            alert_recorded_for_episode = True

                        # ------------------------------------
                        # ALARM WITH COOLDOWN
                        # ------------------------------------

                        if (
                            time.time()
                            -
                            last_alarm
                            >= ALARM_COOLDOWN
                        ):

                            threading.Thread(
                                target=play_alarm,
                                daemon=True
                            ).start()

                            last_alarm = time.time()

                        # Keep longest duration updated.
                        longest_drowsiness = max(
                            longest_drowsiness,
                            closed_time
                        )

            # ------------------------------------------------
            # EYES OPEN
            # ------------------------------------------------

            else:

                open_frames += 1

                closed_frames = 0

                status = "EYES OPEN"

                if open_frames >= 3:

                    closed_start = None

                    # New closed-eye episode can now
                    # generate a new alert.
                    alert_recorded_for_episode = False

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        else:

            latest_ear = 0

            closed_start = None

            closed_frames = 0

            open_frames = 0

            alert_recorded_for_episode = False

        current_status = status

        # ----------------------------------------------------
        # CAMERA TEXT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"EAR: {latest_ear:.2f}",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        status_color = (
            (0, 255, 0)
            if status == "EYES OPEN"
            else (0, 165, 255)
        )

        cv2.putText(
            frame,
            "STATUS: " + status,
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        # ----------------------------------------------------
        # ALERT TEXT
        # ----------------------------------------------------

        if status == "DROWSINESS ALERT":

            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (25, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # ----------------------------------------------------
        # SEND FRAME
        # ----------------------------------------------------

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if ret:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def login():

    if "logged_in" in session:

        return redirect(
            url_for("driver_profile")
        )

    error = ""

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if verify_user(
            username,
            password
        ):

            session["logged_in"] = True

            session["username"] = username

            return redirect(
                url_for("driver_profile")
            )

        error = "Invalid username or password"

    return render_template(
        "login.html",
        error=error
    )


# ============================================================
# CREATE ACCOUNT
# ============================================================

@app.route(
    "/create_account",
    methods=["GET", "POST"]
)
def create_account():

    if load_user():

        return redirect(
            url_for("login")
        )

    error = ""

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        confirm = request.form.get(
            "confirm_password",
            ""
        ).strip()

        if len(username) < 3:

            error = (
                "Username must be at least 3 characters."
            )

        elif len(password) < 4:

            error = (
                "Password must be at least 4 characters."
            )

        elif password != confirm:

            error = (
                "Passwords do not match."
            )

        else:

            save_user(
                username,
                password
            )

            return redirect(
                url_for("login")
            )

    return render_template(
        "create_account.html",
        error=error
    )


# ============================================================
# DRIVER PROFILE
# ============================================================

@app.route(
    "/driver_profile",
    methods=["GET", "POST"]
)
def driver_profile():

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )

    error = ""

    if request.method == "POST":

        name = request.form.get(
            "driver_name",
            ""
        ).strip()

        age = request.form.get(
            "age",
            ""
        ).strip()

        license_number = request.form.get(
            "license_number",
            ""
        ).strip()

        vehicle_number = request.form.get(
            "vehicle_number",
            ""
        ).strip()

        vehicle_type = request.form.get(
            "vehicle_type",
            ""
        ).strip()

        if not all([
            name,
            age,
            license_number,
            vehicle_number,
            vehicle_type
        ]):

            error = (
                "Please fill all the details."
            )

        else:

            data = {
                "driver_name": name,
                "age": age,
                "license_number": license_number,
                "vehicle_number": vehicle_number,
                "vehicle_type": vehicle_type
            }

            save_driver(data)

            return redirect(
                url_for("dashboard")
            )

    return render_template(
        "driver_profile.html",
        profile=load_driver(),
        error=error
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )

    profile = load_driver()

    if not profile:

        return redirect(
            url_for("driver_profile")
        )

    return render_template(
        "index.html",
        profile=profile
    )


# ============================================================
# SESSION REPORT
# ============================================================

@app.route("/session_report")
def session_report():

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )

    return render_template(
        "session_report.html"
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# START MONITORING
# ============================================================

@app.route(
    "/start_monitoring",
    methods=["POST"]
)
def start_monitoring():

    global monitoring_active
    global monitoring_start_time
    global total_monitoring_seconds
    global latest_ear

    if "logged_in" not in session:

        return jsonify(
            {"error": "Unauthorized"}
        ), 401

    start_session()

    monitoring_active = True

    monitoring_start_time = time.time()

    total_monitoring_seconds = 0

    latest_ear = 0

    return jsonify({
        "success": True,
        "active": True
    })


# ============================================================
# STOP MONITORING
# ============================================================

@app.route(
    "/stop_monitoring",
    methods=["POST"]
)
def stop_monitoring():

    global monitoring_active
    global total_monitoring_seconds

    if "logged_in" not in session:

        return jsonify(
            {"error": "Unauthorized"}
        ), 401

    if monitoring_active:

        total_monitoring_seconds += (
            time.time()
            -
            monitoring_start_time
        )

    monitoring_active = False

    return jsonify({
        "success": True,
        "active": False
    })


# ============================================================
# SYSTEM STATS
# ============================================================

@app.route("/system_stats")
def system_stats():

    if "logged_in" not in session:

        return jsonify(
            {"error": "Unauthorized"}
        ), 401

    if monitoring_active:

        monitoring_time = (
            total_monitoring_seconds
            +
            time.time()
            -
            monitoring_start_time
        )

    else:

        monitoring_time = (
            total_monitoring_seconds
        )

    return jsonify({

        # Current detection
        "ear": round(
            latest_ear,
            2
        ),

        "status": current_status,

        "ear_threshold": EAR_THRESHOLD,

        "drowsiness_time": DROWSINESS_TIME,

        # Monitoring time
        "monitoring_seconds": round(
            monitoring_time,
            1
        ),

        # Compatibility with session report
        "monitoring_time": round(
            monitoring_time,
            1
        ),

        "monitoring_active":
            monitoring_active,

        # Alerts
        "total_alerts":
            alert_count,

        "alert_count":
            alert_count,

        # Session
        "session_start":
            time.strftime(
                "%d-%m-%Y %H:%M:%S",
                time.localtime(
                    session_start
                )
            ),

        "session_duration":
            round(
                monitoring_time,
                1
            ),

        # EAR
        "average_ear":
            round(
                average_ear,
                3
            ),

        # Drowsiness
        "longest_drowsiness":
            round(
                longest_drowsiness,
                1
            ),

        # Safety score
        "safety_score":
            calculate_safety_score()
    })


# ============================================================
# ALERTS
# ============================================================

@app.route("/alerts")
def alerts():

    if "logged_in" not in session:

        return jsonify(
            {"error": "Unauthorized"}
        ), 401

    return jsonify({

        "count":
            alert_count,

        "history":
            alert_history
    })


# ============================================================
# RESET ALERTS
# ============================================================

@app.route(
    "/reset_alerts",
    methods=["POST"]
)
def reset_alerts():

    global alert_count
    global alert_history
    global longest_drowsiness

    if "logged_in" not in session:

        return jsonify(
            {"error": "Unauthorized"}
        ), 401

    alert_count = 0

    alert_history = []

    longest_drowsiness = 0

    return jsonify({
        "success": True
    })


# ============================================================
# RESET / CHANGE PASSWORD
# ============================================================

@app.route(
    "/reset_password",
    methods=["GET", "POST"]
)
def reset_password():

    if request.method == "POST":

        current_password = request.form.get(
            "current_password",
            ""
        )

        new_password = request.form.get(
            "new_password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        user = load_user()

        if not user:

            return render_template(
                "reset_password.html",
                error="No account found."
            )

        if not verify_user(
            user["username"],
            current_password
        ):

            return render_template(
                "reset_password.html",
                error=(
                    "Current password is incorrect."
                )
            )

        if len(new_password) < 4:

            return render_template(
                "reset_password.html",
                error=(
                    "New password must be at least 4 characters."
                )
            )

        if new_password != confirm_password:

            return render_template(
                "reset_password.html",
                error=(
                    "New passwords do not match."
                )
            )

        save_user(
            user["username"],
            new_password
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "reset_password.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    global monitoring_active

    monitoring_active = False

    release_camera()

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
from ultralytics import YOLO
import cv2
import time
import numpy as np


# LOAD YOLO MODEL
model = YOLO("models/yolov8n.pt")

# OPEN VIDEO
cap = cv2.VideoCapture("videos/road.mp4")


# PREVIOUS LEFT LANE VALUES
prev_left_x1 = 0
prev_left_y1 = 0
prev_left_x2 = 0
prev_left_y2 = 0

# PREVIOUS RIGHT LANE VALUES
prev_right_x1 = 0
prev_right_y1 = 0
prev_right_x2 = 0
prev_right_y2 = 0


while True:

    # START FPS TIMER
    start_time = time.time()

    # READ FRAME
    ret, frame = cap.read()

    if not ret:
        break

    # RESIZE FRAME
    frame = cv2.resize(frame, (960, 540))


    # ================= YOLO DETECTION =================

    results = model.track(
        frame,
        tracker="bytetrack.yaml"
    )[0]

    filtered_boxes = []

    # FILTER ONLY CAR + PERSON
    for box in results.boxes:

        cls = int(box.cls[0])
        name = model.names[cls]

        if name in ["car", "person"]:
            filtered_boxes.append(box)

    results.boxes = filtered_boxes

    # DRAW YOLO BOXES
    annotated_frame = results.plot()


    # ================= COLLISION WARNING =================

    frame_height = frame.shape[0]

    for box in filtered_boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        #BOX CENTER
        box_center_x = (x1 + x2)//2

        #FRAME CENTER
        frame_center = frame.shape[1] // 2


        object_height = y2 - y1

        # OBJECT TOO CLOSE

        distance = int(5000/object_height)
        


        cv2.putText(
                annotated_frame,
                f"{distance} m",
                (int(x1 + 5), int(y2 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        
        if (
            object_height > frame_height * 0.35
            and abs(box_center_x - frame_center) < 120
            
        ):
            
           


            cv2.putText(
                annotated_frame,
                "WARNING",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


    # ================= LANE DETECTION =================

    # GRAYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # WHITE COLOR FILTER
    _,  white_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    

    # EDGE DETECTION
    edges = cv2.Canny(white_mask, 70, 180)

    # ROI MASK
    height, width = edges.shape[:2]

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (150,  height),           # sol alt  → (150, 540)
        (810,  height),           # sağ alt  → (810, 540)
        (600,  int(height*0.68)), # sağ üst  → (600, 367)
        (360,  int(height*0.68)), # sol üst  → (360, 367)
]], np.int32)


    cv2.fillPoly(mask, polygon, 255)

    roi = cv2.bitwise_and(edges, mask)


    # ================= HOUGH LINE DETECTION =================

    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        40,
        minLineLength=80,
        maxLineGap=100
    )




    # ================= LEFT / RIGHT CLASSIFICATION =================

    left_line = []
    right_line = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            # AVOID DIVISION BY ZERO
            if x2 - x1 == 0:
                continue

            # LINE SLOPE
            slope = (y2 - y1) / (x2 - x1)

            # IGNORE HORIZONTAL + EXTREME LINES
            if abs(slope) < 0.3 or abs(slope) > 2:
                continue

            # spatial filtering
            if x1 > frame.shape[1] * 0.98:
                continue

            # region filtering
            if y1 < frame.shape[0] * 0.5:
                continue



            mid_x = frame.shape[1] // 2

            # LEFT LANE
            if slope < 0 and x1 < mid_x:
                left_line.append(line)

            # RIGHT LANE

            elif slope > 0.25 and x1 > mid_x:
               
                if frame.shape[1] * 0.50 < x1 < frame.shape[1] * 0.98:
               
                   if y1 > frame.shape[0] * 0.35:
                      right_line.append(line)
            
               
            
        

    # ================= LEFT LANE FITTING =================

    left_x = []
    left_y = []

    for line in left_line:

        x1, y1, x2, y2 = line[0]

        left_x.extend([x1, x2])
        left_y.extend([y1, y2])

    if len(left_x) > 0:

        # LINE FITTING
        left_fit = np.polyfit(left_y, left_x, 1)

        # START / END HEIGHT
        y1 = frame.shape[0]
        y2 = int(frame.shape[0] * 0.8)

        # CALCULATE X VALUES
        x1 = int(left_fit[0] * y1 + left_fit[1])
        x2 = int(left_fit[0] * y2 + left_fit[1])

        # TEMPORAL SMOOTHING
        left_x1 = int(0.85 * prev_left_x1 + 0.15 * x1)
        left_y1 = int(0.85 * prev_left_y1 + 0.15 * y1)

        left_x2 = int(0.85 * prev_left_x2 + 0.15 * x2)
        left_y2 = int(0.85 * prev_left_y2 + 0.15 * y2)

        # UPDATE PREVIOUS VALUES
        prev_left_x1 = left_x1
        prev_left_y1 = left_y1

        prev_left_x2 = left_x2
        prev_left_y2 = left_y2

        # DRAW LEFT LANE
        cv2.line(
            annotated_frame,
            (left_x1, left_y1),
            (left_x2, left_y2),
            (255, 0, 0),
            5
        )


    # ================= RIGHT LANE FITTING =================

    right_x = []
    right_y = []

    for line in right_line:
        x1, y1, x2, y2 = line[0]
        right_x.extend([x1, x2])
        right_y.extend([y1, y2])


        # LINE FITTING
        right_fit = np.polyfit(right_y, right_x, 1)

        # START / END HEIGHT
        y1 = frame.shape[0]
        y2 = int(frame.shape[0] * 0.8)

        # CALCULATE X VALUES
        x1 = int(right_fit[0] * y1 + right_fit[1])
        x2 = int(right_fit[0] * y2 + right_fit[1])

        # TEMPORAL SMOOTHING
        right_x1 = int(0.85 * prev_right_x1 + 0.15 * x1)
        right_y1 = int(0.85 * prev_right_y1 + 0.15 * y1)

        right_x2 = int(0.85 * prev_right_x2 + 0.15 * x2)
        right_y2 = int(0.85 * prev_right_y2 + 0.15 * y2)

        # UPDATE PREVIOUS VALUES
        prev_right_x1 = right_x1
        prev_right_y1 = right_y1

        prev_right_x2 = right_x2
        prev_right_y2 = right_y2

        # DRAW RIGHT LANE
        cv2.line(
            annotated_frame,
            (right_x1, right_y1),
            (right_x2, right_y2),
            (0, 255, 0),
            5
        )


    # ================= LANE AREA + STEERING =================

    if len(left_x) > 0 and len(right_x) > 0:

        # POLYGON POINTS
        points = np.array([

            [left_x1, left_y1],
            [left_x2, left_y2],
            [right_x2, right_y2],
            [right_x1, right_y1]

        ])

        # TRANSPARENT OVERLAY
        transparent = annotated_frame.copy()

        # FILL POLYGON
        cv2.fillPoly(
            transparent,
            [points],
            (0, 255, 0)
        )

        # APPLY TRANSPARENCY
        annotated_frame = cv2.addWeighted(
            transparent,
            0.30,
            annotated_frame,
            0.70,
            0
        )

        # LANE CENTER
        lane_center = (left_x1 + right_x1) // 2

        # FRAME CENTER
        offset = 38
        frame_center = frame.shape[1] // 2 + offset

        print("lane_center:", lane_center, "frame_center:", frame_center)


        # ================= LANE DEPARTURE =================

        if abs(lane_center - frame_center) > 60:

            cv2.putText(
                annotated_frame,
                "WARNING: Lane Departure",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        # ================= STEERING ESTIMATION =================

        # TURN LEFT
        if lane_center < frame_center - 80:

            cv2.putText(
                annotated_frame,
                "TURN LEFT",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # TURN RIGHT
        elif lane_center > frame_center + 80:

            cv2.putText(
                annotated_frame,
                "TURN RIGHT",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # CENTER
        else:

            cv2.putText(
                annotated_frame,
                "CENTER",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # ================= FPS =================

    end_time = time.time()

    processing_time = end_time - start_time

    fps = 1 / processing_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # ================= SHOW FRAME =================

    cv2.imshow("Mini_ADAS_System", annotated_frame)


    # EXIT WITH ESC
    if cv2.waitKey(1) == 27:
        break


# RELEASE RESOURCES
cap.release()
cv2.destroyAllWindows()
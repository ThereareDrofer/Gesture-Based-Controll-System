import cv2
import numpy as np

import time
import mediapipe as mp
import math
from datetime import datetime
import matplotlib.pyplot as plt
import simpleaudio as sa
import requests

mp_holistic = mp.solutions.holistic #Holistic model
mp_drawing = mp.solutions.drawing_utils #Drawing utilities

z_list_mp = []
z_list_mp_1 = []
z_list_dist = []
pose = []
pose_new_0 = []
pose_new_1 = []
pose_new_2 = []
pose_new_3 = []
pose_new_4 = []
pose_new_arr = [[], [], [], [], []]

x1_int, y1_int, z1_int = 0, 0, 0
x2_int, y2_int, z2_int = 0, 0, 0
x3_int, y3_int, z3_int = 0, 0, 0
x4_int, y4_int, z4_int = 0, 0, 0

xt, yt, zt = 0, 0, 0
x0, y0, z0 = 0, 0, 0
x1, y1, z1 = 0, 0, 0

def intersect_lines(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    """Определение пересечения двух прямых в пространстве"""
    # задаем направляющие векторы прямых
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x4 - x3, y4 - y3, z4 - z3])
    if np.allclose(np.cross(v1, v2), 0):
        print("Parallels or the same")
        return []
    else:
        try:
            a = np.array([x3 - x1, y3 - y1, z3 - z1])
            M1 = [v1, v2, a]
            b_0 = np.array([v1[1], v1[2]])
            c_0 = np.array([v2[1], v2[2]])
            M2 = [b_0, c_0]
            b_1 = np.array([v1[0], v1[2]])
            c_1 = np.array([v2[0], v2[2]])
            M3 = [b_1, c_1]
            b_2 = np.array([v1[0], v1[1]])
            c_2 = np.array([v2[0], v2[1]])
            M4 = [b_2, c_2]
            distance1 = np.linalg.det(M1) / math.sqrt(
                np.linalg.det(M2) ** 2 + np.linalg.det(M3) ** 2 + np.linalg.det(M4) ** 2)
            distance1 = abs(distance1)
            s0 = (z3 - z1 - (z2 - z1) * (x3 - x1) / (x2 - x1)) / ((z2 - z1) * (x4 - x3) / (x2 - x1) - z4 + z3)
            t0 = ((x4 - x3) * s0 + x3 - x1) / (x2 - x1)
            x_t0 = int((x2 - x1) * t0 + x1)
            y_t0 = int((y2 - y1) * t0 + y1)
            z_t0 = int((z2 - z1) * t0 + z1)
            if distance1 < 20:
                return [x_t0, y_t0, z_t0]
            else:
                return []
        except:
            return []


            # print("t0_x=", t0)
            # x_s0 = int((x4 - x3) * s0 + x3)
            # t0 = ((y4 - y3) * s0 + y3 - y1) / (y2 - y1)
            # print("t0_y=", t0)
            # t0 = ((z4 - z3) * s0 + z3 - z1) / (z2 - z1)
            # print("t0_z=", t0)
            # y_s0 = int((y4 - y3) * s0 + y3)
            # z_s0 = int((z4 - z3) * s0 + z3)

            # distance2 = math.sqrt((x_s0 - x_t0) ** 2 + (y_s0 - y_t0) ** 2 + (z_s0 - z_t0) ** 2)
            # mid_point = [(x_s0 + x_t0) / 2, (y_s0 + y_t0) / 2, (y_s0 + y_t0) / 2]

            # print("Coordinates x_s0... and x_t0... :")
            # print(x_s0, y_s0, z_s0)
            # print(x_t0, y_t0, z_t0)
            #
            # print("Длина между полученными точками = ", distance2)
            # print("Расстояние между прямыми расчитанное по формуле", distance1)
            # print("Координаты точки середины отрезка", mid_point)



def find_point_projection(x0, y0, z0, x1, y1, z1, xt, yt, zt, alfa):
    # координаты точки в трехмерном пространстве
    point = np.array([xt, yt, zt])
    # координаты двух точек, через которые проходит прямая
    line_point_1 = np.array([x0, y0, z0])
    line_point_2 = np.array([x1, y1, z1])
    # вычисляем вектор направления прямой
    line_direction = line_point_2 - line_point_1
    # вычисляем проекцию точки на прямую
    np_dot_1 = np.dot(point - line_point_1, line_direction)
    np_dot_2 = np.dot(line_direction, line_direction)
    projection = line_point_1 + np_dot_1 * line_direction/np_dot_2
    dist_btw_proj_top = math.sqrt((projection[0] - x0) ** 2 + (projection[1] - y0) ** 2 + (projection[2] - z0) ** 2)
    r = math.tan(math.radians(alfa))*dist_btw_proj_top
    dist_btw_proj_point = math.sqrt((projection[0] - xt) ** 2 + (projection[1] - yt) ** 2 + (projection[2] - zt) ** 2)
    if dist_btw_proj_point <= r:
        return True
    else:
        return False

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def arrays_create(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros(33*4)

    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(21*3)

    pose_world = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_world_landmarks.landmark]) if results.pose_landmarks else np.zeros(33 * 4)

    return pose, lh, rh, pose_world


def get_dist_to_cam(Hpx):
    d = 251000/(Hpx+160) - 180
    return d

def play_note():
    # Path to file
    f_name = 'note_2.wav'
    wave_obj = sa.WaveObject.from_wave_file(f_name)

    # Audio playback
    play = wave_obj.play()

    # To stop after playing the whole audio
    play.stop()

def play_note_2():
    # Path to file
    f_name = 'pisk.wav'
    wave_obj = sa.WaveObject.from_wave_file(f_name)

    # Audio playback
    play = wave_obj.play()

    # To stop after playing the whole audio
    play.stop()

def model_gesture(hand):
    fin_straight = []
    direct = []
    dist_from_edge_fin = math.sqrt((hand[mp_holistic.HandLandmark][4][0] -
                                    hand[mp_holistic.HandLandmark][17][0]) ** 2 +
                                   (hand[mp_holistic.HandLandmark][4][1] -
                                    hand[mp_holistic.HandLandmark][17][1]) ** 2)
    dist_from_bones = math.sqrt((hand[mp_holistic.HandLandmark][2][0] -
                                 hand[mp_holistic.HandLandmark][17][0]) ** 2 +
                                (hand[mp_holistic.HandLandmark][2][1] -
                                 hand[mp_holistic.HandLandmark][17][1]) ** 2)
    if hand[mp_holistic.HandLandmark][0][1] < hand[mp_holistic.HandLandmark][4][1]:
        direct.append(0)
    else:
        direct.append(1)

    if dist_from_edge_fin > dist_from_bones:
        fin_straight.append(1)
    else:
        fin_straight.append(0)

    for k in range(5, 18, 4):
        dist_from_edge_fin = math.sqrt((hand[mp_holistic.HandLandmark][k + 3][0] -
                                        hand[mp_holistic.HandLandmark][0][0]) ** 2 +
                                       (hand[mp_holistic.HandLandmark][k + 3][1] -
                                        hand[mp_holistic.HandLandmark][0][1]) ** 2)
        dist_from_bones = math.sqrt((hand[mp_holistic.HandLandmark][k][0] -
                                     hand[mp_holistic.HandLandmark][0][0]) ** 2 +
                                    (hand[mp_holistic.HandLandmark][k][1] -
                                     hand[mp_holistic.HandLandmark][0][1]) ** 2)
        if dist_from_bones < dist_from_edge_fin:
            fin_straight.append(1)
        else:
            fin_straight.append(0)

    if sum(fin_straight) == 1:
        if fin_straight.index(1) == 0:
            if direct[0] == 0:
                return 7
            if direct[0] == 1:
                return 6
        else:
            return (fin_straight.index(1) + 1)
    else:
        if sum(fin_straight) == 5:
            return 0
        else:
            return None

def turn_on():
    esp32_ip = "192.168.200.100"
    resp = requests.get(f"http://{esp32_ip}:8080/diod?signal=1", verify=False)
    print("Turn on")


def turn_off():
    esp32_ip = "192.168.200.100"
    resp = requests.get(f"http://{esp32_ip}:8080/diod?signal=0", verify=False)
    print("Turn off")

p_time = time.time()
start_time = time.time()
rtsp_url = "rtsp://192.168.200.1"
cap = cv2.VideoCapture(rtsp_url)
frame_count = 0
z_final_arr = []
x_final_arr = []
x_final_plus = 0
z_final_plus = 0
inter_point_arr = []
x1_int, y1_int, z1_int = 0, 0, 0
x2_int, y2_int, z2_int = 0, 0, 0
x3_int, y3_int, z3_int = 0, 0, 0
x4_int, y4_int, z4_int = 0, 0, 0
time_start = time.time()
time_start_2 = time.time()

# Set mediapipe model

with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    while cap.isOpened():
        frame_count = frame_count + 1
        timer = int(time.time() - start_time)
        # Read feed
        ret, image = cap.read()

        # Make detections
        image, results = mediapipe_detection(image, holistic)

        imgH, imgW, imgC = image.shape

        arrays_holistic = arrays_create(results)
        x0_world, y0_world, z0_world = 0, 0, 0
        x1_world, y1_world, z1_world = 0, 0, 0

        x0, y0, z0 = 0, 0, 0
        x1, y1, z1 = 0, 0, 0

        x_or = 0
        x_or_arr = []

        dist_to_cam = 0

        try:
            if arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_WRIST.value][3] > 0.8:
                if arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_ELBOW.value][3] > 0.8:
                    x0_world = arrays_holistic[3][mp_holistic.PoseLandmark.RIGHT_ELBOW.value][0]
                    y0_world = arrays_holistic[3][mp_holistic.PoseLandmark.RIGHT_ELBOW.value][1]
                    z0_world = arrays_holistic[3][mp_holistic.PoseLandmark.RIGHT_ELBOW.value][2]
                    x0_world = int(x0_world*100)
                    y0_world = int(y0_world*100)
                    z0_world = int(z0_world*100)

                    x1_world = arrays_holistic[3][mp_holistic.PoseLandmark.RIGHT_WRIST.value][0]
                    y1_world = arrays_holistic[3][mp_holistic.PoseLandmark.RIGHT_WRIST.value][1]
                    z1_world = arrays_holistic[3][mp_holistic.PoseLandmark.RIGHT_WRIST.value][2]

                    x1_world = int(x1_world*100)
                    y1_world = int(y1_world*100)
                    z1_world = int(z1_world*100)

                    x0 = int(arrays_holistic[0][mp_holistic.PoseLandmark.LEFT_HIP.value][0]*1000)
                    y0 = int(imgH - arrays_holistic[0][mp_holistic.PoseLandmark.LEFT_HIP.value][1]*1000)
                    z0 = int(arrays_holistic[0][mp_holistic.PoseLandmark.LEFT_HIP.value][2]*1000)


                    x1 = int(arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_HIP.value][0]*1000)
                    y1 = int(imgH - arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_HIP.value][1]*1000)
                    z1 = int(arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_HIP.value][2]*1000)

                    x_or = (x0 + x1)/2

                    y_right_shoulder = arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_SHOULDER.value][1]
                    y_left_shoulder = arrays_holistic[0][mp_holistic.PoseLandmark.LEFT_SHOULDER.value][1]
                    y_right_hip = arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_HIP.value][1]
                    y_left_hip = arrays_holistic[0][mp_holistic.PoseLandmark.LEFT_HIP.value][1]
                    Hpx = abs((y_right_shoulder + y_left_shoulder)/2 - (y_right_hip + y_left_hip)/2) * imgH

                    dist_to_cam = int(get_dist_to_cam(Hpx))

                    if x_or < 500:
                        k = (500-x_or)/500
                        tg_30 = math.sqrt(3)/3
                        x_final_plus = (-1)*int(k*tg_30*dist_to_cam/(math.sqrt((k*tg_30)**2 + 1)))
                        z_final_plus = int(dist_to_cam/(math.sqrt((k*tg_30)**2 + 1)))

                    if x_or > 500:
                        k = (1000-x_or)/500
                        tg_30 = math.sqrt(3)/3
                        x_final_plus = int(k*tg_30*dist_to_cam/(math.sqrt((k*tg_30)**2 + 1)))
                        z_final_plus = int(dist_to_cam / (math.sqrt((k * tg_30) ** 2 + 1)))

                    if x_or == 500:
                        x_final_plus = 0
                        z_final_plus = dist_to_cam




            # cv2.putText(image, "x0 = " + str(x0), (40, 60),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # cv2.putText(image, "y0 = " + str(y0), (40, 100),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # cv2.putText(image, "z0 = " + str(z0), (40, 140),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # cv2.putText(image, "x1 = " + str(x1), (40, 180),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # cv2.putText(image, "y1 = " + str(y1), (40, 220),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # cv2.putText(image, "z1 = " + str(z1), (40, 260),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # cv2.putText(image, "Distance to camera= " + str(dist_to_cam) + " cm", (40, 330),
            #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
            # cv2.putText(image, "X_OR = " + str(x_or), (40, 500),
            #             cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 10)

        except:
            pass

        if (time.time() - time_start) > 15 and x1_int == 0:
            cv2.putText(image, "INDEX FIRST LINE!", (800, 200),
                        cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 10)

        if x1_int != 0 and len(inter_point_arr) == 0:
            cv2.putText(image, "INDEX SECOND LINE!", (800, 200),
                        cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 10)

        if arrays_holistic[1].all() != 0 and (time.time() - time_start) > 15 and len(pose) == 0:
            if model_gesture(arrays_holistic[1]) == 0:
                if arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_ELBOW.value][3] > 0.8 \
                        and arrays_holistic[0][mp_holistic.PoseLandmark.RIGHT_WRIST.value][3] > 0.8:
                    try:
                        time_start = time.time()
                        pose = arrays_holistic[0]
                        pose_new_0 = arrays_holistic[3]

                        x1_int, y1_int, z1_int = int(x0_world + x_final_plus), 100 - y0_world, int(
                            z0_world + z_final_plus)
                        x2_int, y2_int, z2_int = int(x1_world + x_final_plus), 100 - y1_world, int(
                            z1_world + z_final_plus)
                        z_final_arr.append(z_final_plus)
                        x_final_arr.append(x_final_plus)
                        play_note()

                        if len(z_final_arr) != 0:
                            cv2.putText(image, "FIRST LINE IS GOOD", (800, 600),
                                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

                    except:
                        pass


        if len(inter_point_arr) == 0 and len(pose) != 0:
            cv2.putText(image, "NOT Found point of intersection", (800, 300),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
            if (time.time() - time_start) > 5:
                if arrays_holistic[1].all() != 0:
                    if model_gesture(arrays_holistic[1]) == 0:
                        # Нахождение пересечения прямых
                        x3_int, y3_int, z3_int = int(x0_world + x_final_plus), 100 - y0_world, int(
                            z0_world + z_final_plus)
                        x4_int, y4_int, z4_int = int(x1_world + x_final_plus), 100 - y1_world, int(
                            z1_world + z_final_plus)
                        print("Finding")
                        print("Coordinates of first line:")
                        print("(", x1_int, ",", y1_int, ",", z1_int, ")")
                        print("(", x2_int, ",", y2_int, ",", z2_int, ")")
                        print("Coordinates of second line:")
                        print("(", x3_int, ",", y3_int, ",", z3_int, ")")
                        print("(", x4_int, ",", y4_int, ",", z4_int, ")")
                        inter_point_arr = intersect_lines(x1_int, y1_int, z1_int, x2_int, y2_int, z2_int,
                                                          x3_int, y3_int, z3_int, x4_int, y4_int, z4_int)

        if len(inter_point_arr) != 0 and len(pose_new_1) == 0:
            time_start_2 = time.time()
            pose_new_1 = arrays_holistic[3]
            z_final_arr.append(z_final_plus)
            x_final_arr.append(x_final_plus)
            play_note()

        if len(inter_point_arr) != 0 and len(pose_new_1) != 0:
            cv2.putText(image, "SUCCESS!!!", (1000, 100),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

        if (time.time() - time_start_2) > 5 and len(inter_point_arr) != 0:
            # Нахождение точки в конусе
            x1_int, y1_int, z1_int = int(x0_world + x_final_plus), 100 - y0_world, int(z0_world + z_final_plus)
            x2_int, y2_int, z2_int = int(x1_world + x_final_plus), 100 - y1_world, int(z1_world + z_final_plus)
            xt, yt, zt = inter_point_arr[0], inter_point_arr[1], inter_point_arr[2]

            if find_point_projection(x1_int, y1_int, z1_int, x2_int, y2_int, z2_int, xt, yt, zt, 10):
                cv2.putText(image, "Zone of element", (1000, 300),
                            cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
                play_note_2()
                if (model_gesture(arrays_holistic[2]) == 6):
                    turn_on()
                if (model_gesture(arrays_holistic[2]) == 7):
                    turn_off()

        ###########################################
        """Сканирование расстояния до человека для построения функции"""
        # if 60 <= frame_count < 70:
        #     cv2.putText(image, "Prepare", (40, 500),
        #                 cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
        #
        # if 70 <= frame_count < 180:
        #     cv2.putText(image, "Scaning", (40, 500),
        #                 cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

            # z_list_mp.append(x1_world)
            # z_list_mp_1.append(y1_world)
            # z_list_dist.append(dist_to_cam)

        #############################################################################################
        """Прошлая часть"""
        """
        if 60 <= frame_count < 120:
            cv2.putText(image, "Prepare to index first element", (800, 200),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            z_list_mp.append(x1_world)
            z_list_mp_1.append(y1_world)
            z_list_dist.append(dist_to_cam)

        if frame_count == 121:
            try:
                if len(pose) == 0:
                    pose = arrays_holistic[0]
                    pose_new_0 = arrays_holistic[3]

                    x1_int, y1_int, z1_int = int(x0_world + x_final_plus), 100 - y0_world, int(z0_world + z_final_plus)
                    x2_int, y2_int, z2_int = int(x1_world + x_final_plus), 100 - y1_world, int(z1_world + z_final_plus)
                    print("Coordinates of first line:")
                    print("(", x1_int, ",", y1_int, ",", z1_int, ")")
                    print("(", x2_int, ",", y2_int, ",", z2_int, ")")
                    z_final_arr.append(z_final_plus)
                    x_final_arr.append(x_final_plus)
                    play_note()


            except:
                pass

        if 121 < frame_count < 150:
            if len(z_final_arr) != 0:
                cv2.putText(image, "Coordinates #1 was got!", (800, 200),
                           cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 10)
            else:
                cv2.putText(image, "Coordinates #1 WAS NOT got!", (800, 200),
                            cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 10)

        if len(inter_point_arr) == 0:
            cv2.putText(image, "NOT Found point of intersection", (800, 300),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
            if frame_count > 200:
                # Нахождение пересечения прямых
                x3_int, y3_int, z3_int = int(x0_world + x_final_plus), 100 - y0_world, int(z0_world + z_final_plus)
                x4_int, y4_int, z4_int = int(x1_world + x_final_plus), 100 - y1_world, int(z1_world + z_final_plus)
                print("Finding")
                print("Coordinates of second line:")
                print("(", x3_int, ",", y3_int, ",", z3_int, ")")
                print("(", x4_int, ",", y4_int, ",", z4_int, ")")
                inter_point_arr = intersect_lines(x1_int, y1_int, z1_int, x2_int, y2_int, z2_int,
                                                  x3_int, y3_int, z3_int, x4_int, y4_int, z4_int)


        if len(inter_point_arr) != 0 and len(pose_new_1) == 0:
            pose_new_1 = arrays_holistic[3]
            z_final_arr.append(z_final_plus)
            x_final_arr.append(x_final_plus)
            play_note()

        if len(inter_point_arr) != 0 and len(pose_new_1) != 0:
            cv2.putText(image, "Found point of intersection", (800, 300),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

        if 150 <= frame_count < 240:
            cv2.putText(image, "Prepare to index second element", (800, 220),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
        if frame_count == 241:
            try:
                # pose_new_1 = arrays_holistic[3]
                # z_final_arr.append(z_final_plus)
                # x_final_arr.append(x_final_plus)
                play_note()
            except:
                pass
        if frame_count > 300 and len(inter_point_arr) != 0:
            # Нахождение точки в конусе
            x1_int, y1_int, z1_int = int(x0_world + x_final_plus), 100 - y0_world, int(z0_world + z_final_plus)
            x2_int, y2_int, z2_int = int(x1_world + x_final_plus), 100 - y1_world, int(z1_world + z_final_plus)
            xt, yt, zt = inter_point_arr[0], inter_point_arr[1], inter_point_arr[2]

            if find_point_projection(x1_int, y1_int, z1_int, x2_int, y2_int, z2_int, xt, yt, zt, 10):
                cv2.putText(image, "Zone of element", (800, 1000),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                play_note_2()


        if 241 < frame_count < 300:
            cv2.putText(image, "Coordinates #2 was got!", (800, 200),
                        cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 10)
        if 300 <= frame_count < 360:
            cv2.putText(image, "Prepare to index third element", (800, 200),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
        if frame_count == 361:
            try:
                pose_new_2 = arrays_holistic[3]
                z_final_arr.append(z_final_plus)
                x_final_arr.append(x_final_plus)
                play_note()
            except:
                pass

        if 361 < frame_count < 450:
            cv2.putText(image, "Coordinates #3 was got!", (800, 200),
                        cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 10)

        if frame_count == 550:
            try:
                pose_new_3 = arrays_holistic[3]
            except:
                pass

        if frame_count == 550:
            try:
                pose_new_4 = arrays_holistic[3]
            except:
                pass
        
        

        cv2.putText(image, "xt = " + str(xt), (800, 80),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.putText(image, "yt = " + str(yt), (800, 120),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.putText(image, "zt = " + str(zt), (800, 160),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.putText(image, "Current time:", (1000, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.putText(image, str(datetime.now().strftime("%H:%M:%S")), (1000, 80),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.putText(image, "Timer: "+str(30 - timer), (1000, 120),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        """
        ##################################################

        c_time = time.time()
        fps = int(1/(c_time-p_time))
        p_time = c_time
        cv2.putText(image, "FPS " + str(fps), (10, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('MediaPipe output', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # plt.plot(z_list_mp, 'red')
    # plt.plot(z_list_mp_1, 'orange')
    # plt.plot(z_list_dist, 'black')


    pose_new_arr = pose_new_0, pose_new_1, pose_new_2, pose_new_3, pose_new_4

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = 'r', 'g', 'b', 'y', 'black'
    for p in range(3):
        pose_new = pose_new_arr[p]
        x_mas = []
        y_mas = []
        z_mas = []
        try:
            for i in range(33):
                x_mas.append(int(pose_new[i][0] * 100)+x_final_arr[p])
                y_mas.append(int((1 - pose_new[i][1]) * 100))
                z_mas.append(int(pose_new[i][2] * 100)+z_final_arr[p])
        except:
            pass
        if len(x_mas) != 0:
            ax.scatter(x_mas, z_mas, y_mas, c=colors[p], s=20)

            new_x_mas = []
            new_y_mas = []
            new_z_mas = []

            d = math.sqrt((x_mas[16] - x_mas[14]) ** 2 + (y_mas[16] - y_mas[14]) ** 2 + (z_mas[16] - z_mas[14]) ** 2)
            L = 600
            x3_line = x_mas[16] + L * (x_mas[16] - x_mas[14]) / d
            y3_line = y_mas[16] + L * (y_mas[16] - y_mas[14]) / d
            z3_line = z_mas[16] + L * (z_mas[16] - z_mas[14]) / d

            for i in range(35):
                new_x_mas.append(x_mas[list(mp_holistic.POSE_CONNECTIONS)[i][0]])
                new_x_mas.append(x_mas[list(mp_holistic.POSE_CONNECTIONS)[i][1]])
                # print(new_x_mas)
                new_y_mas.append(y_mas[list(mp_holistic.POSE_CONNECTIONS)[i][0]])
                new_y_mas.append(y_mas[list(mp_holistic.POSE_CONNECTIONS)[i][1]])
                # print(new_y_mas)
                new_z_mas.append(z_mas[list(mp_holistic.POSE_CONNECTIONS)[i][0]])
                new_z_mas.append(z_mas[list(mp_holistic.POSE_CONNECTIONS)[i][1]])
                # print(new_z_mas)

                ax.plot(new_x_mas, new_z_mas, new_y_mas, colors[p])

                new_x_mas = []
                new_y_mas = []
                new_z_mas = []

            ax.plot([x3_line, x_mas[16]], [z3_line, z_mas[16]], [y3_line, y_mas[16]], colors[p])

    ax.scatter(inter_point_arr[0], inter_point_arr[2], inter_point_arr[1], s=20, c='black')
    # ax.plot([x1_int, inter_point_arr[0]], [z1_int, inter_point_arr[2]], [y1_int, inter_point_arr[1]], c='black')
    ax.plot([x1_int, x2_int], [z1_int, z2_int], [y1_int, y2_int], c='black')
    ax.plot([x3_int, x4_int], [z3_int, z4_int], [y3_int, y4_int], c='orange')
    # ax.plot([x3_int, inter_point_arr[0]], [z3_int, inter_point_arr[2]], [y3_int, inter_point_arr[1]], c='black')

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    plt.xlim([-200, 200])
    plt.ylim([0, 400])

    plt.show()

    cap.release()
    cv2.destroyAllWindows()



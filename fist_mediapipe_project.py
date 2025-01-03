import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Получение размеров экрана
root = tk.Tk()
screen_width = 2 * root.winfo_screenwidth()
screen_height = 2 * root.winfo_screenheight()
root.destroy()

# Скачивание картинок, распознование их размеров
imageH = cv2.imread('imageVi.png')
imageJ = cv2.imread('imageJa.png')
imageA = cv2.imread('imageUsa.png')
imageF = cv2.imread('imageFr.png')

heighth, widthh, _ = imageH.shape
heightj, widthj, _ = imageJ.shape
heighta, widtha, _ = imageA.shape
heightf, widthf, _ = imageF.shape
heightf, widthf, _ = imageF.shape


# Создаем переменные с текстом
intro_text = "Hi, dear friend!"
introo_text = "To travel to a country, repeat the movement of the choosed country)"
introoo_text = "To start, click 'space'"
introooo_text = "To exit, press 'esc'"

rect = 1

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)

# Установка разрешения видео в Full HD
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Устанавливаем размер окна на 80% от размера экрана
window_width = int(screen_width * 0.9)
window_height = int(screen_height * 0.9)

tip_ids = [4, 8, 12, 16, 20] # Кисть
base_ids = [0, 5, 9, 13, 17] # Кончики пальцев
joint_ids = [3, 6, 10, 14, 18] # Часть фаланг

# Пороговые значения углов для пальцев
thumb_bend_threshold = 40
finger_bend_threshold = 80


def get_angle(v1, v2): # Возвращает угол между двумя векторами
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def is_finger_bent(base, joint, tip, is_thumb=False): #Определяет согнут ли палец
    v1 = [joint.x - base.x, joint.y - base.y, joint.z - base.z] # Вектор между основание и суставом
    v2 = [tip.x - joint.x, tip.y - joint.y, tip.z - joint.z] # Вектор между суставом и концом
    angle = get_angle(v1, v2) # Считаем угол между векторами
    if is_thumb:
        return angle < thumb_bend_threshold
    else:
        return angle < finger_bend_threshold

step = 0

backgrounds = {
    "USA": "USA_photo.png",
    "Vietnam": "Vietnam_photo.png",
    "France": "France_photo.png",
    "Japan": "Japan_photo.png",
}

current_background = None  # По умолчанию естественный фон (None)
background = None  # Начальная переменная для фона

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.fliplr(frame)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Сегментация кадра
    segmentation_results = selfie_segmentation.process(frame)
    mask = segmentation_results.segmentation_mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    binary_mask_inv = 1 - binary_mask


    if results.multi_hand_landmarks: # Если обнаружена рука
        for hand_landmarks in results.multi_hand_landmarks: # Идем по каждой точке
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Рисуем все точки и линии
            landmarks = hand_landmarks.landmark

            for finger_index, tip_id in enumerate(tip_ids): # Идем по кончикам
                base_id = base_ids[finger_index] # Находим основание данного пальца
                joint_id = joint_ids[finger_index] # Находим сустав данного пальца
                is_thumb = (finger_index == 0) #Определение большого пальца
                if is_finger_bent(landmarks[base_id], landmarks[joint_id], landmarks[tip_id], is_thumb): # Если палец согнут, то красный круг, иначе зеленый
                    cx, cy = int(landmarks[tip_id].x * frame.shape[1]), int(landmarks[tip_id].y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                    if finger_index == 0:
                        thumbbent = True
                    elif finger_index == 1:
                        ukazbent = True
                    elif finger_index == 2:
                        sredbent = True
                    elif finger_index == 3:
                        bezbent = True
                    else:
                        mizinbent = True

                else:
                    cx, cy = int(landmarks[tip_id].x * frame.shape[1]), int(landmarks[tip_id].y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                    if finger_index == 0:
                        thumbbent = False
                    elif finger_index == 1:
                        ukazbent = False
                    elif finger_index == 2:
                        sredbent = False
                    elif finger_index == 3:
                        bezbent = False
                    else:
                        mizinbent = False
        country = ''
        if thumbbent and not ukazbent and not sredbent and not bezbent and not mizinbent:
            country = 'USA'
        elif not thumbbent and ukazbent and sredbent and not bezbent and not mizinbent:
            country = 'Vietnam'
        elif not thumbbent and not ukazbent and sredbent and bezbent and mizinbent:
            country = 'France'
        elif not thumbbent and not ukazbent and not sredbent and not bezbent and mizinbent:
            country = 'Japan'
        else:
            # Сбрасываем фон, если жест не распознан
            current_background = None
            background = None

        # Замена фона
        if country and backgrounds.get(country) and backgrounds[country] != current_background:
            current_background = backgrounds[country]
            background = cv2.imread(current_background)
            if frame.shape[:2] != background.shape[:2]:
                background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

        if background is not None:
            person = cv2.bitwise_and(frame, frame, mask=binary_mask)
            new_background = cv2.bitwise_and(background, background, mask=binary_mask_inv)
            frame = cv2.add(person, new_background)


    if step > 0:
        frame[500:500 + heighth, 100:100 + widthh] = imageH
        frame[500:500 + heightj, 395:395 + widthj] = imageJ
        frame[500:500 + heighta, 690:690 + widtha] = imageA
        frame[500:500 + heightf, 985:985 + widthf] = imageF

    if cv2.waitKey(1) & 0xFF == ord(' '):
        step+=1
        intro_text, introo_text, introoo_text, introooo_text = "", "", "", ""
        rect = 0

    if rect == 1:
        cv2.rectangle(frame, (0,0), (2000,2000), (0,0,0), thickness=-1)
    cv2.putText(frame, str(intro_text), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), thickness=2)
    cv2.putText(frame, str(introo_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), thickness=2)
    cv2.putText(frame, str(introoo_text), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), thickness=2)
    cv2.putText(frame, str(introooo_text), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), thickness=2)
    # Масштабируем изображение до размера окна
    frame_resized = cv2.resize(frame, (window_width, window_height))

    cv2.imshow('Fingers', frame_resized)
    cv2.resizeWindow('Fingers', window_width, window_height)

    if cv2.waitKey(10) == 27:
        break


cap.release()
cv2.destroyAllWindows()
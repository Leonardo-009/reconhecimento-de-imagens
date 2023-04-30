import random

import cv2
import numpy as np
from ultralytics import YOLO

# Ira abri o arquivo e ler o modo
my_file = open("G:/reconhecimento-de-imagens/yolov8n/coco.names", "r")
# esta lendo arquivo
data = my_file.read()
# substituindo end dividindo o texto | quando a nova linha ('\n') é vista.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Ira gera cores aleatórias para a lista de classes
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# carrega um modelo YOLOv8n pré-treinado
model = YOLO("G:/reconhecimento-de-imagens/yolov8n/yolov8n.pt", "v8")

#Valores para redimensionar quadros de vídeo | quadros pequenos otimizam a execução
frame_wid = 1920
frame_hyt = 1080

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Não é possível abrir a câmera")
    exit()

while True:
    # Captura frame a frame
    ret, frame = cap.read()
    # ser o if for lido corretamente, ret será True.

    if not ret:
        print("Não é possível receber o quadro (fim do fluxo?). Saindo...")
        break

    # Previa de uma imagem
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Converter uma array  para numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # ira retonar uma box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            #Ira Exibir o nome da classe
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Ira retonra o nome do Display
    cv2.imshow("Objeto Detectado", frame)

    # presioner x para termina a execução do projeto
    if cv2.waitKey(1) == ord("x"):
        break

# ser tudo corre bem ira libera o captura a img
cap.release()
cv2.destroyAllWindows()
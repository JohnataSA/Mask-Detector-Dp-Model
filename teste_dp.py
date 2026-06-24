from ultralytics import YOLO
import cv2

# Carrega o modelo pré-treinado YOLOv8 (pode ser 'yolov8n.pt' para mais leve)
model = YOLO("mask_v2.pt")

# Testar via Device ou Video
#cap = cv2.VideoCapture(0)
video_path = "cap2.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    # Redimensiona o frame para o tamanho da tela
    resized_frame = cv2.resize(annotated_frame, (1500, 900), interpolation=cv2.INTER_AREA)

    # Exibe o frame na janela
    cv2.imshow("Detecção Dp John", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
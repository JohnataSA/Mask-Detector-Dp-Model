from ultralytics import YOLO
import cv2

# Carrega o modelo pré-treinado YOLOv8 (pode ser 'yolov8n.pt' para mais leve)
model = YOLO("mask_v1.pt")

# Abre o vídeo (substitua pelo caminho do seu vídeo)
#video_path = "cap.mp4"
#cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a detecção
    results = model(frame)

    # Desenha as detecções no frame

    annotated_frame = results[0].plot()

    # Reduz o tamanho do frame para a janela (ajuste `SCALE` conforme necessário)
    SCALE = 0.6  # valor entre 0.1 (muito pequeno) e 1.0 (tamanho original)
    h, w = annotated_frame.shape[:2]
    new_w = max(1, int(w * SCALE))
    new_h = max(1, int(h * SCALE))
    resized_frame = cv2.resize(annotated_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Permite redimensionamento da janela se o usuário quiser arrastar
    cv2.namedWindow("Detecção Dp John", cv2.WINDOW_NORMAL)
    cv2.imshow("Detecção Dp John", resized_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import yaml



#Nova maneira de definir pesos no Ultralytics# Define pesos para cada classe (exemplo com 3 classes)# Pesos mais altos = mais importância na perda
#class_weights = torch.tensor([1.0, 1.0, 3.0, 2.0])  # ajuste conforme necessário# Substitui os pesos no loss
#model.model.loss.class_weights = class_weights

def main():
    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("fine_v1.pt")  # load a pretrained model (recommended for training)

    # Use the model
    #model.train(data="capacete.yaml", epochs=30, device=0)  # train the model
    model.train(data="mask.yaml", epochs=60, batch=16, lr0=0.001, lrf=0.01, freeze="backbone",
        augment=True, patience=15, cos_lr=True, workers=4, imgsz=640, name="fine_tune_mask_v3", save=True,)  # train the modelel
    
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print("path", path)


if __name__ == '__main__':
    # freeze_support()
    main()

    # fine_tune.py
#from ultralytics import YOLO
# Carrega o modelo pré-treinado
#model = YOLO("mask_v1.pt")  # Seu modelo atual

# Configurações de treinamento
# #train_args = {
#     "data": "mask.yaml",          # Arquivo de configuração
#     "epochs": 100,                # Número de épocas
#     "batch": 16,                  # Tamanho do batch (reduza se der OOM)
#     "lr0": 0.001,                 # Learning rate inicial (baixo para fine-tuning)
#     "lrf": 0.01,                  # Learning rate final
#     "freeze": "backbone",         # Congela as camadas iniciais (transfer learning)
#     "augment": True,              # Habilita augmentations do Ultralytics
#     "patience": 20,               # Early stopping (para se não melhorar em 20 épocas)
#     "class_weights": [1.0, 1.0, 3.0, 2.0],  # Peso para classes (mask_weared_incorrect = 3.0)
#     "cos_lr": True,               # Learning rate dinâmico (melhora convergência)
#     "device": 0,                  # Usa GPU (se disponível)
#     "workers": 4,                 # Número de workers para carregar dados
#     "imgsz": 640,                 # Tamanho da imagem (padrão YOLOv8)
#     "name": "fine_tune_mask_v2",  # Nome da pasta de saída (em runs/detect/)
# }

# Inicia o treinamento
#results = model.train(**train_args)

# Salva o modelo treinado
#model.save("mask_v2.pt")  # Novo modelo após fine-tuning
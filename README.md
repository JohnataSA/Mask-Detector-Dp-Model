#Treino de Deep Learning para detecção do uso de máscaras faciais

Informações úteis:

1. Dataset coletado do Kaggle: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/code
2. Anotações feitas com o LabelImg.
3. Base utilizada: Yolo26n.
4. Yaml de configuração: mask.yaml
5. Arquivo para realizar o treinamento: train.v8.py
6. Arquivo para realizar os testes de detecção: teste_dp.py
7. DeepLearning: mask.v1.pt

Resultados:

<img width="1288" height="632" alt="image" src="https://github.com/user-attachments/assets/c65ad36b-ab2f-42b3-8577-cecab7167e07" />

Matriz de confusão:

<img width="859" height="631" alt="image" src="https://github.com/user-attachments/assets/96cfeb3a-6e9e-47b7-9440-715c0e1d1f34" />

Teste de Detecção:

<img width="1276" height="337" alt="image" src="https://github.com/user-attachments/assets/5a7d977f-61d7-46b6-9d32-824fbd5625ed" />

Conclusões desta fase:

O modelo tem um bom desempenho para as classes with_mask e without_mask, mas precisa de melhorias para:

mask_weared_incorrect (acurácia de apenas 15%).
background (acurácia de 21%).

Próxima fase:
Aumentar o dataset para as classes com baixa acurácia.
Revisar as imagens de fundo para evitar falsos positivos.
Ajustar o threshold de detecção para reduzir confusões.



#Treino de Deep Learning para detecção do uso de máscaras faciais

Informações úteis:

1. Dataset coletado do Kaggle: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/code
2. Anotações feitas com o LabelImg.
3. Base utilizada: Yolo26n.
4. Yaml de configuração: mask.yaml
5. Arquivo para realizar o treinamento: train.v8.py
6. Arquivo para realizar os testes de detecção: teste_dp.py
7. DeepLearning: mask.v1.pt
8. Final-DeepLearning: mask.v2.pt

Resultados do primeiro Treinamento (30 Epochs):

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

#######################################################
#                   AFTER FINE-TUNING                 #
#######################################################

Resultados do Segundo Treinamento, Fine Tuning (40 Epochs):

<img width="1289" height="641" alt="image" src="https://github.com/user-attachments/assets/22c629a0-fdc1-4579-a9f4-1018c7a8aa73" />

Matriz de confusão 2:

<img width="733" height="644" alt="image" src="https://github.com/user-attachments/assets/3affc1cf-c98a-4f54-a90b-ca6fd562d973" />

Teste de Detecção:

<img width="1142" height="291" alt="image" src="https://github.com/user-attachments/assets/6f601f06-f977-4d1d-aa61-d1dc118dde4b" />

Conclusões Final:

mAP50: 0.8856
Precision: 0.8667
Recall: 0.8652
F1-Score: 0.8659

Após fine tuning, o modelo se comportou da maneira esperada resolvendo consideralvente o problema de detecção de posição incorreta da máscara.
Antes cerca de apenas 15% dessa classe era detectada, e após a 2 etapa de treino passou a detectar 78% da classe.
Foi adicionado mais imagens deste case e utilizado técnica de data argumentation como complemento do resultado final.

#######################################################
#                        THE END                      #
#######################################################






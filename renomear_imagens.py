import os

# Caminho da pasta contendo as imagens
pasta = "C:/Users/Johna/Downloads/archive/dataset3"

# Prefixo desejado
prefixo = "maksssksksss1001"

# Contador para numerar os arquivos
contador = 1

# Lista de extensões de imagem comuns
extensoes_imagem = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# Verifica se a pasta existe
if not os.path.exists(pasta):
    print(f"Pasta não encontrada: {pasta}")
else:
    # Itera sobre os arquivos na pasta
    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.lower().endswith(extensoes_imagem):
            # Cria o novo nome do arquivo
            novo_nome = f"{prefixo}{contador}{os.path.splitext(nome_arquivo)[1]}"

            # Caminho completo do arquivo antigo e novo
            caminho_antigo = os.path.join(pasta, nome_arquivo)
            caminho_novo = os.path.join(pasta, novo_nome)

            # Renomeia o arquivo
            os.rename(caminho_antigo, caminho_novo)
            print(f"Renomeado: {nome_arquivo} -> {novo_nome}")

            # Incrementa o contador
            contador += 1

    print("Renomeação concluída!")
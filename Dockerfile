
# Usa uma imagem Python oficial como base
FROM python:3.9-slim



# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr
# Define o diretório de trabalho dentro do container
WORKDIR /app




# Copia os arquivos de requirements para dentro do container
COPY app/requirements.txt .

# Instala as dependências da API
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da API
COPY app/ .

# Expondo a porta da API
EXPOSE 5000

# Comando para iniciar a API
CMD ["python", "main.py"]
    
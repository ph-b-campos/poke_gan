# Ambiente para treinar e avaliar as GANs. Mesma versão de Python do desenvolvimento (3.14).
# No Linux o torch do PyPI já traz CUDA: com o NVIDIA Container Toolkit, `docker run --gpus all` usa a GPU;
# sem ele, roda em CPU (lento para treinar, suficiente para gerar figuras de um modelo já treinado).
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# dados/ e resultados/ ficam fora da imagem: monte-as como volumes (ver README)
CMD ["python", "-m", "data_handler", "--baixar"]

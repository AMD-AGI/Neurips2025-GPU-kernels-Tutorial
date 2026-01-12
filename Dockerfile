FROM rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.8.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/jupyter/

COPY src /app/jupyter/src/

# Clone GEAK-agent@neurips for Main.ipynb (src/geak/)
RUN rm -rf /app/jupyter/src/geak/GEAK-agent && \
    git clone -b neurips --depth 1 https://github.com/AMD-AGI/GEAK-agent.git /app/jupyter/src/geak/GEAK-agent || \
    git clone --depth 1 https://github.com/AMD-AGI/GEAK-agent.git /app/jupyter/src/geak/GEAK-agent

# Clone GEAK-agent@geak-openevolve for GEAK-evolve-handson.ipynb (src/geak-evolve/)
RUN rm -rf /app/jupyter/src/geak-evolve/GEAK-openevolve && \
    git clone -b geak-openevolve --depth 1 https://github.com/AMD-AGI/GEAK-agent.git /app/jupyter/src/geak-evolve/GEAK-openevolve || \
    git clone --depth 1 https://github.com/AMD-AGI/GEAK-agent.git /app/jupyter/src/geak-evolve/GEAK-openevolve

# Clone GEAK-eval@geak-oe for evaluation data (used by GEAK-evolve-handson.ipynb)
RUN git clone -b geak-oe --depth 1 https://github.com/AMD-AGI/GEAK-eval.git /app/jupyter/src/geak-evolve/GEAK-openevolve/GEAK-eval-OE || \
    git clone --depth 1 https://github.com/AMD-AGI/GEAK-eval.git /app/jupyter/src/geak-evolve/GEAK-openevolve/GEAK-eval-OE

# Install dependencies for both notebooks
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir ipykernel jupyterlab && \
    if [ -s /app/jupyter/src/geak/requirements.txt ]; then pip install --no-cache-dir -r /app/jupyter/src/geak/requirements.txt; fi && \
    if [ -s /app/jupyter/src/geak/GEAK-agent/requirements.txt ]; then pip install --no-cache-dir -r /app/jupyter/src/geak/GEAK-agent/requirements.txt; fi && \
    if [ -s /app/jupyter/src/geak-evolve/GEAK-openevolve/requirements.txt ]; then pip install --no-cache-dir -r /app/jupyter/src/geak-evolve/GEAK-openevolve/requirements.txt; fi && \
    pip install --no-cache-dir -e /app/jupyter/src/geak-evolve/GEAK-openevolve/ && \
    pip install --no-cache-dir -e /app/jupyter/src/geak-evolve/GEAK-openevolve/GEAK-eval-OE/ --no-deps && \
    rm -rf /root/.cache/pip

EXPOSE 8888 8000 30000

ENV JUPYTER_TOKEN="neurips2025"
ENV JUPYTER_BASE_URL="/"

CMD ["/bin/bash", "-lc", "\
    jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --allow-root \
    --ServerApp.token=${JUPYTER_TOKEN} \
    --ServerApp.base_url=${JUPYTER_BASE_URL} \
    --ServerApp.open_browser=False \
    --ServerApp.trust_xheaders=True \
    --ServerApp.allow_origin='*' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_remote_access=True \
    --ServerApp.allow_credentials=True \
    --notebook-dir=/app/jupyter"]
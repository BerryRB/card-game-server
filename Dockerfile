# =============================================================
# 升级(Trump)纸牌游戏服务器 — 含DMC V9 AI (CPU版)
# =============================================================
# 构建镜像:
#   docker build -t trump-game-server .
#
# 运行（模型通过挂载提供）:
#   docker run -d -p 9999:9999 \
#     -v /path/to/models_v9:/app/rl_dmc/models_v9:ro \
#     --name trump-server trump-game-server
#
# 运行（模型已打包进镜像）:
#   # 先把模型放到 rl_dmc/models_v9/ 下，然后:
#   docker build -t trump-game-server .
#   docker run -d -p 9999:9999 trump-game-server
#
# 运行测试:
#   docker run --rm trump-game-server pytest test_dmc_unit.py test_dmc_e2e.py -v
# =============================================================

# ---- Stage 1: 安装Python依赖 ----
FROM python:3.11-slim AS builder

WORKDIR /build

# 先装依赖（利用Docker层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ---- Stage 2: 运行时镜像 ----
FROM python:3.11-slim

LABEL maintainer="BerryRB"
LABEL description="升级(Trump)纸牌游戏服务器 — DMC V9 AI (CPU)"

WORKDIR /app

# 从builder拷贝已安装的Python包（仅CPU版torch，约200MB）
COPY --from=builder /install /usr/local

# 拷贝应用代码
COPY server/         server/
COPY rl_dmc/         rl_dmc/
COPY rl_shengji/     rl_shengji/
COPY static/         static/
COPY templates/      templates/
COPY app.py          .
COPY requirements.txt .

# 模型文件(~70MB)提供方式:
# 方式1: 构建时放入 rl_dmc/models_v9/ 后COPY进来（见下方注释）
# 方式2: 运行时 -v 挂载（推荐，镜像更小更通用）
#
# 若选择方式1，取消下方注释:
# COPY rl_dmc/models_v9/ rl_dmc/models_v9/

# 拷贝测试文件
COPY test_dmc_unit.py    .
COPY test_dmc_e2e.py     .
COPY test_web_e2e_v2.py  .

# 非root用户运行
RUN groupadd -r trump && useradd -r -g trump -d /app trump \
    && chown -R trump:trump /app
USER trump

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9999/')" || exit 1

EXPOSE 9999

CMD ["python", "app.py"]

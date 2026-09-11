# AGENTS.md — Heimdall Gateway (llamacpp-stack)

> Guía para agentes de codificación (LLMs) que trabajan en este repositorio. Lee este archivo antes de explorar, instalar, testear o modificar código.

## 1. Qué es este repo

**Heimdall Gateway** es un gateway de inferencia local, compatible con OpenAI, para ejecutar y rotar LLMs en hardware propio. Orquesta:

- Descarga y registro de modelos GGUF de Hugging Face (selección de quant, shards).
- Generación de `config.yaml` para `llama-swap` desde un catálogo (`catalog.json`) + ajustes globales (`conf.json`).
- Ruteo OpenAI-compatible hacia `llama.cpp` (`llama-server`) o backend `vLLM` (beta).
- Réplicas conversation-affine, auto-context, MTP/speculative, defaults por familia, placement GPU, logs/diagnóstico y cleanup seguro de huérfanos.

Comando público único: **`heimdall-gateway`** (`llamacpp_stack/llamacpp_api_install.py:main` -> `pyproject.toml:15`). No hay alias legacy.

Arquitectura resumida:

```
cliente OpenAI --> Heimdall API :11435 --> llama-swap :11436 --> llama-server (GGUF) / vllm-server (HF)
                                      --> conf.json + catalog.json --> config.yaml (generado)
```

Detalles completos en `README.md` y `docs/LLM_INSTALL.md` (checklist obligatorio para installs asistidos).

---

## 2. Regla de oro: `uv` es el gestor preferente

> **En este repositorio se usa preferentemente `uv` para todo lo que tiene que ver con instalaciones en este entorno o en instaladores.**

Esto aplica a:

- Crear entornos virtuales.
- Instalar dependencias del proyecto y de desarrollo.
- Instalaciones editables (`pip install -e .`).
- Invocar tests, linters y el CLI en el entorno del repo.
- Cualquier script o automatización que instale paquetes Python.

**No uses `pip`/`pip3`/`python -m pip` ni `venv` puro si `uv` está disponible.** Si `uv` no está instalado, instálalo primero y luego continúa. Los ejemplos de `README.md` que usan `pip` son válidos como fallback documentado, pero para trabajo de agente en este repo debes traducirlos a equivalentes `uv`.

### Equivalencias rápidas

| Acción | Con `pip` (README) | Con `uv` (preferido aquí) |
|---|---|---|
| Crear venv Python 3.12 | `python3.12 -m venv .venv` | `uv venv --python 3.12 .venv` |
| Activar venv | `source .venv/bin/activate` | `source .venv/bin/activate` (igual) |
| Instalar proyecto editable | `python -m pip install -e .` | `uv pip install -e .` o `uv sync` |
| Instalar proyecto no editable | `python -m pip install .` | `uv pip install .` |
| Sincronizar lockfile | — | `uv sync` (lee `uv.lock` + `pyproject.toml`) |
| Añadir dependencia | `pip install <pkg>` | `uv add <pkg>` / `uv pip install <pkg>` |
| Correr tests | `pytest -q` | `uv run pytest -q` |
| Correr CLI | `heimdall-gateway info` | `uv run heimdall-gateway info` |
| Correr script Python | `python tools/foo.py` | `uv run python tools/foo.py` |

> Si necesitas reproducir exactamente un comando del `README.md` o de `docs/LLM_INSTALL.md` para verificar documentación, puedes mostrar ambas variantes, pero ejecuta la variante `uv`.

### Por qué `uv.lock` manda

- El repo fija `requires-python >=3.12` y el lock en `uv.lock` (versión actual con `huggingface-hub==1.10.1`, `pyyaml`, `requests`, `hf-transfer`, etc.).
- `uv sync` respeta `uv.lock` y es más rápido/determinista que `pip install`.
- En instaladores/automatización, preferir `uv pip install` mantiene el entorno reproducible.

### Nota para `install.py` / bundle

- `llamacpp_stack/install.py` y `llamacpp_stack/bundle/install_llamacpp_stack.sh` son los instaladores de **usuario final** del gateway (servicios systemd, modelos, certs). Crean su propio bootstrap venv si hace falta.
- Cuando como agente **desarrollas o testeas** esos instaladores, usa `uv` para preparar el entorno de desarrollo desde el que los invocas (`uv run heimdall-gateway install --mode user --backend auto --dry-run`), no para reemplazar la lógica interna del instalador salvo que el cambio solicitado sea precisamente migrar el instalador a `uv`.

---

## 3. Requisitos

- **Linux** (única plataforma soportada para gateway).
- **Python >=3.12** (ver `pyproject.toml:6`).
- `uv` instalado (preferente). Fallback: `python3.12` + `pip >= 23`.
- NVIDIA driver/CUDA solo si se usa inferencia GPU; el gateway en sí funciona en CPU.
- Espacio en disco para modelos (ej. Qwen 32B Q4 ~18 GB).
- `cmake`, `ninja`, `curl`, `git` si compilas `llama.cpp` en modo `source`.

Verificación rápida del entorno (ejecuta antes de cualquier install):

```bash
uv --version 2>&1 || echo "uv no instalado"
uv python list 2>&1 | head -n 20
python3.12 --version 2>&1; which python3.12
nvidia-smi 2>&1 | head -n 30 || echo "sin GPU/driver"
df -h /var /home 2>&1 | head -n 20
ss -tlnp | grep -E '11434|11435|11436' || echo "puertos libres"
heimdall-gateway info 2>&1 | head -n 80 || echo "gateway no instalado"
```

---

## 4. Mapa del repositorio

```
.
├── llamacpp_stack/          # Paquete principal (heimdall-gateway)
│   ├── cli.py               # CLI y help epilog
│   ├── install.py           # Instalador user/system, prompts, resolve_* (fuente de verdad)
│   ├── llamacpp_api_install.py  # Entry point `heimdall-gateway`
│   ├── command_router.py    # Ruteo API -> llama-swap
│   ├── managed_commands.py  # Comandos `add`/`run`/`update`/`validate`
│   ├── auto_performance.py / auto_perf_runner.py
│   ├── dependencies.py      # Chequeo deps
│   ├── bundle/              # install_llamacpp_stack.sh, common.sh, defaults YAML
│   │   └── llama_server_defaults.yaml
│   └── migrations/          # Migraciones de conf.json
├── configs/history/         # Snapshots de autotuning
├── docs/
│   ├── LLM_INSTALL.md       # Guía canónica para LLMs (Q1-Q8, matriz, comandos)
│   ├── VLLM-BETA.md
│   ├── LOCAL_OLLAMA_SETUP.md
│   ├── arg-hyphen-conventions.md
│   └── flags_llamacpp
├── tests/                   # pytest (ver §7)
├── tools/                   # bench/verify scripts
├── skills/heimdall-autotune/ # Skill de autotuning
├── templates/               # Templates de config
├── Dockerfile / Dockerfile.llamacpp / Dockerfile.vllm
├── docker-compose-vllm.yaml
├── pyproject.toml           # deps proyecto: hf-transfer, huggingface-hub, pyyaml, requests, optuna
├── uv.lock                  # lockfile determinista (usar con `uv sync`)
├── requirements.txt         # mirror mínimo de deps (legacy, preferir pyproject/uv)
└── README.md
```

Ficheros de **instalación real** (no editar `config.yaml` a mano):

| Propósito | user mode | system mode |
|---|---|---|
| Settings globales | `~/.config/heimdall-gateway/conf.json` | `/etc/heimdall-gateway/conf.json` |
| Catálogo modelos | `~/.local/state/heimdall-gateway/catalog.json` | `/var/lib/heimdall-gateway/catalog.json` |
| Runtime generado | `~/.local/state/heimdall-gateway/config.yaml` | `/var/lib/heimdall-gateway/config.yaml` |
| Env wrappers | `~/.config/heimdall-gateway/heimdall-gateway.env` | `/etc/heimdall-gateway/heimdall-gateway.env` |
| Request log | `~/.local/state/heimdall-gateway/api-requests.log` | `/var/lib/heimdall-gateway/api-requests.log` |

---

## 5. Setup de desarrollo (con `uv`)

```bash
# 0. uv (si falta)
curl -LsSf https://astral.sh/uv/install.sh | sh
# o: pipx install uv

# 1. Clonar y entrar
git clone <repo> && cd llamacpp-stack

# 2. Crear venv y sincronizar deps (determinista)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync
# alternativa editable explícita:
uv pip install -e .

# 3. Verificar CLI
uv run heimdall-gateway --help
uv run heimdall-gateway info
uv run heimdall-gateway config-keys --format json | head -n 80

# 4. Tests
uv run pytest -q

# 5. Instalación local de prueba (no toca sistema, sin servicios)
uv run heimdall-gateway install --mode user --backend auto \
  --llama-cpp-mode prebuilt \
  --models-dir /tmp/heimdall-models \
  --public-host 127.0.0.1 --public-port 11436 \
  --no-api-auth --no-api-https --idle-ttl 300 \
  --no-install-services --dry-run
```

> Para desarrollo con checkout sin `pip install` previo, el wrapper `./llamacpp_stack/bundle/install_llamacpp_stack.sh` sigue funcionando (crea bootstrap venv). Si lo usas desde un entorno `uv`, invócalo como `uv run bash llamacpp_stack/bundle/install_llamacpp_stack.sh ...`.

---

## 6. Flujo de instalación asistida (para agentes)

Si el usuario te pide instalar el gateway, sigue **estrictamente** `docs/LLM_INSTALL.md`:

1. Ejecuta pre-checks (§1 de ese doc) y guarda outputs.
2. Haz **Q1–Q8 en un solo bloque** (modo, models-dir, backend, método llama.cpp, red/puertos, auth/TLS, modelos iniciales, tuning). No re-preguntes lo ya respondido. En non-TTY usa defaults + flags explícitos.
3. Ejecuta siempre primero con `--dry-run`, revisa el plan, luego sin `--dry-run`.
4. Verifica con `heimdall-gateway info` + `curl /v1/models` + `curl /api/replicas`.
5. No asumas `sudo`, puertos o descargas de GB sin confirmar.

Comandos canónicos (variante `uv`):

```bash
# Nueva instalación user, auto, interactiva mínima
uv run heimdall-gateway install --mode user --backend auto --models-dir /var/llamacpp_models --idle-ttl 300 --dry-run
uv run heimdall-gateway install --mode user --backend auto --models-dir /var/llamacpp_models --idle-ttl 300

# System (re-ejecuta con sudo -E interno si hace falta)
uv run heimdall-gateway install --mode system --backend auto --models-dir /var/llamacpp_models --public-host 0.0.0.0 --public-port 11436

# No interactivo / CI
uv run python -m llamacpp_stack.install --mode user --backend auto --llama-cpp-mode prebuilt \
  --models-dir /tmp/heimdall-models --public-host 127.0.0.1 --public-port 11436 \
  --no-api-auth --no-api-https --idle-ttl 300 --no-install-services --dry-run
```

---

## 7. Tests y validación

```bash
# Todos (recomendado con uv)
uv run pytest -q
uv run pytest tests/test_managed_commands.py -q
uv run pytest tests/test_llamacpp_install.py -q

# Con verbose / filtro
uv run pytest -v -k "replica or config_migrate"

# Verificación de comandos llama.cpp generados
uv run python tools/verify_llama_cmds.py
```

Notas:
- `pyproject.toml:28-30` define `pythonpath = ["."]` y `testpaths = ["tests"]`.
- Muchos tests usan fixtures/mocks para evitar GPU/descargas reales. Cambios que afecten hardware deben validarse en instalación real.
- Antes de push, revisa `git diff --cached` y evita commitear secretos.

---

## 8. Docker

```bash
docker build -t heimdall-gateway -f Dockerfile .
docker build -t heimdall-llamacpp -f Dockerfile.llamacpp .
docker build -t heimdall-vllm -f Dockerfile.vllm .
docker compose -f docker-compose-vllm.yaml up --build
```

---

## 9. Configuración

- **No edites `config.yaml` a mano** — se regenera desde `conf.json` + `catalog.json` en cada `update`.
- Tras editar `conf.json` o `catalog.json`:

  ```bash
  uv run heimdall-gateway config-migrate
  uv run heimdall-gateway update
  systemctl --user restart heimdall-gateway-manager heimdall-gateway-router
  # system: sudo systemctl restart heimdall-gateway-manager heimdall-gateway-router
  ```

- Inspección de claves soportadas:

  ```bash
  uv run heimdall-gateway config-keys
  uv run heimdall-gateway config-keys --format json
  uv run heimdall-gateway info
  uv run heimdall-gateway hacks
  ```

- Defaults tunables: `llamacpp_stack/bundle/llama_server_defaults.yaml` (sampling, KV-cache, batching, GPU, context).

---

## 10. Convenciones y estilo

- Python 3.12, `setuptools` (ver `pyproject.toml:18-19`).
- Formato de claves: ver `docs/arg-hyphen-conventions.md` — distinguir `snake_case` en JSON vs `kebab-case` en CLI/flags.
- Mantener idempotencia en `install.py` (`detect_existing_mode`, `choose_default_swap_port`, preservación de `models-dir`/certs/puertos en updates).
- Mensajes de instalador: preguntar antes de escribir en disco, usar `sudo`, abrir puertos o descargar modelos.
- Commits: mensajes concisos, sin secretos, con `git status`/`git diff` revisados.

---

## 11. Troubleshooting (resumen)

- **API 502 / Connection refused :11436**: `llama-server` crasheó en load (OOM, ctx excesivo, draft MTP incompatible). `uv run heimdall-gateway logs --lines 200 --journal` + `nvidia-smi`.
- **Setting no visible**: confirmar `info` muestra el `mode` correcto, luego `config-migrate && update && restart`.
- **GPU equivocada**: inspeccionar `CUDA_VISIBLE_DEVICES`/`--device`/`--tensor-split` en `ps`/`logs`.
- **Contexto pequeño en cliente**: `curl /v1/models | jq` tiene el real; el cliente puede tener metadata cacheada.

Ver `README.md#Troubleshooting` y `docs/LLM_INSTALL.md#8`.

---

## 12. Seguridad

- No commitear `api_key`, claves privadas, certs, `api-requests.log`, `catalog.json` con rutas privadas ni `*.env`.
- Preferir variables de entorno o secretos gestionados por el instalador.
- Bind privado o HTTPS+auth si se expone fuera de red confiable.
- `git diff --cached` + secret scan antes de push a rama pública.

---

## 13. Referencias

- `README.md` — overview, arquitectura, API, troubleshooting.
- `docs/LLM_INSTALL.md` — checklist canónico para agentes (pre-checks, Q1–Q8, matriz, comandos, verificación).
- `llamacpp_stack/install.py` — fuente de verdad de prompts/flags (`prompt_bool`, `prompt_choice`, `resolve_*`, `build_cli_parser`).
- `docs/VLLM-BETA.md`, `docs/LOCAL_OLLAMA_SETUP.md`, `docs/arg-hyphen-conventions.md`, `docs/flags_llamacpp`, `docs/lllamacpp_flags_API.md`.
- `skills/heimdall-autotune/SKILL.md` — loop de autotuning.

---

*Última actualización: 2026-09-04. Si añades una dependencia, usa `uv add` y commitea `uv.lock`.*

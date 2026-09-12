# Heimdall Gateway - Guia de Instalacion para LLMs (Agentes)

> **Audiencia:** Este documento es para **LLMs / agentes de codificacion** que deben instalar y configurar Heimdall Gateway en una maquina del usuario de forma autonoma o semi-asistida. Si eres humano, usa `README.md#Install`.

## 0. Principio operativo

*   **No asumas nada.** Pregunta antes de escribir en disco, usar `sudo`, abrir puertos o descargar modelos (pesan GB).
*   **Se idempotente.** `install --mode X --backend auto` y `install --dry-run` son seguros de re-ejecutar. La instalacion preserva `models-dir`, certs y puertos en updates.
*   **Verifica siempre.** Despues de cada paso corre `heimdall-gateway info` y `curl /v1/models`.

---

## 1. Pre-checks obligatorios (antes de preguntar nada)

Ejecuta y guarda outputs para decidir defaults:

```bash
# 1. OS y arch
uname -a; cat /etc/os-release 2>/dev/null | head -n 20

# 2. Python
python3 --version; python3.12 --version 2>&1; which python3.12

# 3. GPU / driver
nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 || echo "no nvidia-smi"
nvidia-smi 2>&1 | head -n 30
# conteo para presets (llama_server_defaults.yaml: 1/2/4 GPU)
python3 -c "import shutil; print(shutil.which('nvidia-smi'))"

# 4. CUDA toolkit / nvcc
which nvcc; nvcc --version 2>&1 | head -n 5
# tambien via uv/pip
python3 -c "import site,glob,os; print(glob.glob(os.path.join(site.getsitepackages()[0],'nvidia','cuda_nvcc','bin','nvcc')))" 2>&1 | head

# 5. Espacio y puertos
df -h /var /home 2>&1 | head -n 20
ss -tlnp | grep -E '11434|11435|11436' || echo "puertos libres"
systemctl cat ollama 2>&1 | grep -i OLLAMA_HOST | head

# 6. Instalacion existente
heimdall-gateway info 2>&1 | head -n 80
ls -la ~/.config/heimdall-gateway/conf.json /etc/heimdall-gateway/conf.json 2>&1 | head
ls -la ~/.local/state/heimdall-gateway/catalog.json /var/lib/heimdall-gateway/catalog.json 2>&1 | head
cat ~/.config/heimdall-gateway/heimdall-gateway.env 2>&1 | head -n 30
```

Si `heimdall-gateway` no esta en `PATH`, el binario esta en `~/.local/bin/heimdall-gateway` tras `pip install .` o disponible via `./llamacpp_stack/bundle/install_llamacpp_stack.sh`.

## 2. Preguntas clave al usuario (NO instales sin esto)

Haz estas preguntas **en un solo bloque** para minimizar turnos. Si el usuario ya dio alguna, no repitas. Si corre en `non-TTY`, usa defaults y flags explicitos.

### Q1 - Modo de instalacion
> "¿Quieres instalacion **user** (sin sudo, todo en `~/.local/`, `~/.config/heimdall-gateway/`, `~/.local/state/heimdall-gateway/`) o **system** (requiere `sudo`, en `/etc/heimdall-gateway`, `/var/lib/heimdall-gateway`, servicios del sistema)?"
*   Default: `user` si no es `root`; `system` si `EUID==0`.
*   Flag: `--mode user|system`. Si ya existe, el installer mantiene el modo detectado (`detect_existing_mode()` en `install.py:792`).

### Q2 - Directorio de modelos
> "¿Donde guardo los modelos? Ej: `/var/llamacpp_models` (default user: `~/.local/share/heimdall-gateway/models` o `/var/llamacpp_models` si existe). ¿Cuanto espacio tienes? Un Qwen 32B Q4 ~18GB."
*   Flag: `--models-dir /ruta`. Obligatorio si no hay default. El installer pregunta interactivamente si se omite y `stdin.isatty()`.

### Q3 - Backend
> "¿Backend? Recomiendo `auto` (GGUF -> llama.cpp, HF nativo -> vLLM). Opciones legacy `llama.cpp` / `vllm-beta` solo afectan default historico; ambos motores se instalan igual."
*   Flag: `--backend auto` (siempre preferir `auto`).

### Q4 - Metodo llama.cpp
> "¿Como instalo llama.cpp? `source` (compila local, mejor tuning GPU, tarda 10-20 min, necesita `cmake`, `ninja`, `nvcc` si CUDA), `prebuilt` (binario oficial, rapido), `native` (usar paquete del sistema ya instalado)."
*   Default interactivo: `source` (`install.py:812`).
*   Flags: `--llama-cpp-mode source|prebuilt|native` + opcional `--llama-cpp-ref <tag|commit>` (fuerza `source`). Si eliges `commit`, pregunta: "¿Que ref? Ej: `b6408`, `master`." -> `HEIMDALL_GATEWAY_LLAMA_CPP_REF`.

### Q5 - Red / puertos
> "¿Expongo en `127.0.0.1` (solo local) o `0.0.0.0` (LAN)? Puertos: por defecto `ollama_port+2` para llama-swap (`11436` si Ollama en `11434`) y `ollama_port+1` para API Heimdall (`11435`). ¿Tienes Ollama en `11434`? ¿Quieres puertos custom?"
*   Flags: `--public-host 127.0.0.1|0.0.0.0` y `--public-port 11436`.
*   En update, el puerto existente se preserva (`choose_default_swap_port` en `install.py:943`). Solo `--public-port` lo migra explicitamente.

### Q6 - Autenticacion y TLS del API
> "¿Activo auth en el API Heimdall (`:11435`)? Si sí, ¿me das `api-key` o genero una? ¿Activo HTTPS con cert autofirmado? Si tienes cert propio, dame rutas `cert_file`/`key_file`."
*   Flags: `--api-auth/--no-api-auth`, `--api-key <key>`, `--api-https/--no-api-https`, `--api-cert-file`, `--api-key-file`, `--api-cert-sans "myhost.local,10.0.0.5"`, `--regenerate-api-cert` (solo si el usuario lo pide explicitamente).
*   Por defecto el installer genera `api_key` aleatoria si `--api-auth` sin `--api-key` y escribe en `conf.json: api_auth`.

### Q7 - Modelos iniciales (opcional pero recomendado preguntar)
> "¿Quieres que descargue un modelo de prueba tras instalar? Ej: `unsloth/Qwen3.5-32B` o `Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M` (~18GB). Si no, instalo sin modelos."
*   Post-install: `heimdall-gateway add -hf <repo>:<quant>` y `heimdall-gateway run -hf <repo>:<quant> --auto`

### Q8 - Recursos / tuning
> "¿Cuantos GPUs? ¿VRAM por GPU? ¿Quieres tuning agresivo CUDA (`GGML_CUDA_FORCE_MMQ`, `FA_ALL_QUANTS`, `CUBLAS`) o seguro? ¿`idle-ttl` (segundos antes de descargar modelo inactivo, default `300`)?"
*   Flags: `--idle-ttl 300`, env `HEIMDALL_GATEWAY_DISABLE_AGGRESSIVE_CUDA=1` para desactivar, `--prefer-source-cuda/--no-prefer-source-cuda`, `--prefer-binary/--no-prefer-binary`.

> **No preguntes dos veces.** Si el usuario ya respondio `mode=user`, `models-dir=/var/llamacpp_models`, `auto`, etc., construye el comando directamente.

## 3. Matriz de decision rapida

| Senal | Accion LLM |
|---|---|
| `EUID==0` o usuario dice "para todos" | `--mode system` (re-ejecuta con `sudo -E`) |
| `nvidia-smi` falla | CPU-only OK; `source` sin CUDA, avisa que sera lento |
| `nvcc` presente | Sugiere `--prefer-source-cuda` |
| `ollama` activo en `11434` | Deja defaults `11435`/`11436` (ollama+1/+2) |
| `heimdall-gateway info` muestra instalacion existente | Pregunta `full` vs `package-only` (ver `prompt_existing_install_action` en `install.py:775`); usa `--update-binaries/--no-update-binaries` segun respuesta |
| Usuario no interactivo (`!isatty`) | No preguntes interactivamente; usa defaults y flags explicitos |

## 4. Comandos canonicos (copiar/pegar)

### 4.1 Instalacion nueva - usuario, auto, interactiva minima

```bash
# Opcion A: pip + installer (recomendado si ya tienes venv)
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -e .   # o pip install .
heimdall-gateway install --mode user --backend auto --models-dir /var/llamacpp_models --idle-ttl 300 --dry-run
# revisa el plan, luego sin --dry-run
heimdall-gateway install --mode user --backend auto --models-dir /var/llamacpp_models --idle-ttl 300

# Opcion B: wrapper bootstrap (sin pip previo, crea venv solo)
./llamacpp_stack/bundle/install_llamacpp_stack.sh --mode user --backend auto --models-dir /var/llamacpp_models --idle-ttl 300 --dry-run
./llamacpp_stack/bundle/install_llamacpp_stack.sh --mode user --backend auto --models-dir /var/llamacpp_models --idle-ttl 300
```

### 4.2 Instalacion system

```bash
python3.12 -m pip install .
sudo heimdall-gateway install --mode system --backend auto --models-dir /var/llamacpp_models --public-host 0.0.0.0 --public-port 11436
# o con wrapper (el wrapper hace sudo -E interno)
./llamacpp_stack/bundle/install_llamacpp_stack.sh --mode system --backend auto
```

### 4.3 Variantes no interactivas (para LLMs en CI/automatizado)

```bash
# Sin preguntas, todo explicito, sin servicios (para test)
python -m llamacpp_stack.install --mode user --backend auto --llama-cpp-mode prebuilt \
  --models-dir /tmp/heimdall-models --public-host 127.0.0.1 --public-port 11436 \
  --no-api-auth --no-api-https --idle-ttl 300 --no-install-services --dry-run

# Con auth generada y HTTPS autofirmado
heimdall-gateway install --mode user --backend auto --models-dir /var/llamacpp_models \
  --api-auth --api-https --public-host 0.0.0.0 --idle-ttl 300

# Forzar ref concreto de llama.cpp
heimdall-gateway install --mode user --backend auto --llama-cpp-mode source --llama-cpp-ref b10156 --models-dir /var/llamacpp_models
```

### 4.4 Update / reinstalacion (no destructivo)

```bash
heimdall-gateway install --mode user --backend auto --dry-run
heimdall-gateway install --mode user --backend auto
# solo actualizar gateway sin tocar binarios/config
heimdall-gateway install --mode user --backend auto --no-update-binaries --package-only-update
```

## 5. Verificacion post-install (obligatorio para el LLM)

```bash
# 5.1 Servicios
systemctl --user status heimdall-gateway-manager heimdall-gateway-router  # user mode
# o sudo systemctl status heimdall-gateway-manager heimdall-gateway-router  # system mode

# Si estan inactive, arrancar
systemctl --user restart heimdall-gateway-manager heimdall-gateway-router
# o sudo systemctl restart ...

# 5.2 Info agregada (endpoints, versiones, rutas)
heimdall-gateway info
# Debe mostrar: llama.cpp bXXXX, llama-swap vXXX, Install mode, Idle TTL, API/UI reachable

# 5.3 API y UI
curl -s http://127.0.0.1:11435/v1/models | jq '.data[] | {id, context_length}'
curl -s http://127.0.0.1:11436/v1/models | jq '.data | length'
curl -s http://127.0.0.1:11435/api/replicas | jq .

# 5.4 Config y knob logging.llama_swap (verifica que config-keys muestra el knob)
heimdall-gateway config-keys --format json | jq '.logging.llama_swap'
# -> {"logToStdout":"both","logLevel":"info"}  (whitelist both/proxy/upstream/none)
heimdall-gateway config-keys --format json | jq '.logging.requests_log'
# -> {"path": "...", "max_bytes": 10485760, "retain_days": 3, ...}

# Tras cambiar cualquier knob en conf.json, runbook obligatorio:
heimdall-gateway config-migrate && heimdall-gateway update
systemctl --user restart heimdall-gateway-manager heimdall-gateway-router
# system: sudo systemctl restart heimdall-gateway-manager heimdall-gateway-router

# 5.5 Metricas por backend (que esperar, no inventar)
# Llama a un modelo y comprueba timings/usage + Activity:
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"hi"}],"stream":false}' \
  | jq '.timings, .usage'
curl -s http://127.0.0.1:11436/api/metrics/activity | jq '.data[] | {model: .model, tokens: .tokens}'

# Expectativa por backend (ver README Troubleshooting tabla completa):
# - llama.cpp: timings con prompt_ms/predicted_ms + slot print_timing en journal (con logToStdout both/upstream)
# - EXL3: timings reales (prefill_ms/predicted_ms medidos) + draft_n/draft_n_accepted solo con MTP, Cached "-" salvo prefix-cache real, NUNCA slot print_timing (timing vive en timings/Activity)
# - vLLM: prompt/generated nativos en usage, Drafted solo con speculative + flag per_request_spec_decode_metrics, Cached "-", NUNCA slot print_timing

# Verificacion de logs por backend:
heimdall-gateway logs --lines 200 --journal | grep timing
# llama.cpp -> lineas "slot print_timing" presentes con both/upstream
# EXL3/vLLM -> sin print_timing es ESPERADO; buscar "timings" JSON y Activity arriba

# 5.6 Migrate idempotente (segunda pasada no debe cambiar)
heimdall-gateway config-migrate
# -> si ya esta migrado, "already current" / changed==False; segunda pasada identica
# Verificado en tests: logging.llama_swap con valor "verbose" normaliza a both/info y segunda pasada queda estable

# 5.7 Logs si algo falla (rotación 3 días + split + raw ring)
heimdall-gateway logs --lines 200 --journal
heimdall-gateway requests --lines 200
journalctl --user -u heimdall-gateway-manager -n 100 --no-pager
journalctl --user -u heimdall-gateway-router -n 100 --no-pager

# Rotación: api-requests.log.YYYY-MM-DD[.partN] (split por max_bytes, prune >retain_days=3, symlink api-requests.log → día activo)
# Env vars: HEIMDALL_GATEWAY_REQUESTS_LOG_PATH > conf.json logging.requests_log.path > DEFAULT; MAX_BYTES/RETAIN_DAYS vía env o conf.json (clamp 65536-1GiB / 1-30)
# Raw ring: api-raw-requests.log guarda últimas 10 requests RAW sin procesar (1 MiB cap/request, misma resolución de path, fallback /tmp)
```

Si `API status: not reachable`, comprueba `HEIMDALL_GATEWAY_PUBLIC_HOST`/`PORT` en `~/.config/heimdall-gateway/heimdall-gateway.env` y `conf.json`.

### Jinja source (para EXL3/vLLM)

El template viene del repo, no del gateway. Verifica:

```bash
sha256sum /var/llamacpp_models/<model>/chat_template.jinja
python3 -c "import json,hashlib; d=json.load(open('/var/llamacpp_models/<model>/tokenizer_config.json')); print(hashlib.sha256(d['chat_template'].encode()).hexdigest())"
# ambos shas deben coincidir
```

Si difieren, `heimdall-gateway update <id> --auto` re-registra.

## 6. Descargar y probar un modelo (post-install)

```bash
# Registrar sin descargar (solo catalog)
heimdall-gateway add -hf Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M

# Descargar + configurar + calentar (recomendado --auto para autoseleccionar ctx)
heimdall-gateway run -hf Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M --auto

# Verificar
heimdall-gateway list
heimdall-gateway ps
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-32b-instruct-q4_k_m","messages":[{"role":"user","content":"hola"}],"stream":false}' | jq

# MTP / speculative (cuando el repo indica draft)
heimdall-gateway run -hf org/base-model:Q4_K_M --speculative -hf org/draft-model:IQ1_M
```

`--auto` hace probe de ctx real; `--skip-ctx` lo omite. `validate -hf ... --auto` valida sin servir.

## 7. Ficheros clave (no editar `config.yaml` a mano)

| Proposito | user mode | system mode |
|---|---|---|
| Settings globales | `~/.config/heimdall-gateway/conf.json` | `/etc/heimdall-gateway/conf.json` |
| Catalogo modelos | `~/.local/state/heimdall-gateway/catalog.json` | `/var/lib/heimdall-gateway/catalog.json` |
| Runtime generado | `~/.local/state/heimdall-gateway/config.yaml` | `/var/lib/heimdall-gateway/config.yaml` |
| Env wrappers | `~/.config/heimdall-gateway/heimdall-gateway.env` | `/etc/heimdall-gateway/heimdall-gateway.env` |
| Request log (rotado diario, split, 3 días) | `~/.local/state/heimdall-gateway/api-requests.log.YYYY-MM-DD[.partN]` + symlink `api-requests.log` → día activo; prune 3 días; fallback `/tmp/heimdall-gateway-api-requests.log.YYYY-MM-DD` | `/var/lib/heimdall-gateway/api-requests.log.YYYY-MM-DD[.partN]` |
| Raw ring (últimas 10 sin procesar) | `~/.local/state/heimdall-gateway/api-raw-requests.log` (fallback `/tmp/heimdall-gateway-api-raw-requests.log`, 1 MiB cap) | `/var/lib/heimdall-gateway/api-raw-requests.log` |
| Defaults tunables | `llamacpp_stack/bundle/llama_server_defaults.yaml` | mismo |

Editables: `conf.json: llama_server_defaults`, `llama_server_family_defaults`, `replicas`, `experimental`, `api_auth`, `api_https`, `logging.requests_log` (`path`, `max_bytes` 65536-1GiB default 10485760, `retain_days` 1-30 default 3, `compress` bool) y `logging.llama_swap` (`logToStdout` whitelist `proxy|upstream|both|none` default `both`, `logLevel` default `info`). Tras editar: `heimdall-gateway config-migrate && heimdall-gateway update && systemctl --user restart ...`
`config-migrate` añade `logging.requests_log` y `logging.llama_swap` con defaults si falta y nunca sobrescribe valores existentes; segunda pasada `changed==False` idempotente (verificado con `verbose` -> `both` normalizado). `update_config` no reescribe logs. `config-keys` lista `logging` completo, ver con:

```bash
heimdall-gateway config-keys --format json | jq '.logging.llama_swap'
heimdall-gateway config-keys --format json | jq '.logging.requests_log'
```

Verifica metricas por backend con:

```bash
curl -s http://127.0.0.1:11436/api/metrics/activity | jq '.data[] | {model, tokens}'
heimdall-gateway logs --lines 200 --journal | grep timing
# llama.cpp -> slot print_timing presente; EXL3/vLLM -> sin print_timing es esperado, timing en timings/Activity
```

## 8. Errores comunes y que decir al usuario

*   **`502 upstream` / `Connection refused :11436`**: `llama-server` del modelo crasheo en load (OOM, ctx 262k en GPU pequena, draft MTP incompatible). Buscar `bundle_ref` en JSON 502 y `*_with_bundle` en log rotado `api-requests.log.YYYY-MM-DD[.partN]` + `api-raw-requests.log` (últimas 10 RAW, 1 MiB cap, fallback `/tmp`). `uv run heimdall-gateway logs --lines 200 --journal` trae journal + `nvidia-smi`. Bundle caps: journal 12k/nvidia 2k/HTTP 4000.
*   **`model provider failed after retries` (Hermes)**: wrapper generico; ver `requests --lines 200` para `openai_chat_upstream_network_error` / `llamaswap_guard_backend_error`. Casi siempre router caido o modelo no cargado.
*   **`context shown too small`**: `curl /v1/models | jq` tiene el real; cliente cacheo metadata vieja -> refrescar cliente o `update --auto`.
*   **Sin `slot print_timing` en EXL3/vLLM**: es esperado. Solo `llama-server` emite `slot print_timing`. Para EXL3 y vLLM el timing vive en `timings` (`prompt_ms`/`predicted_ms`/`draft_n`) y en `curl :11436/api/metrics/activity`. No reportes como bug si `heimdall-gateway logs --journal | grep timing` no muestra `print_timing` para esos backends.
*   **Metricas `Cached`/`Drafted` en `-`**: es esperado cuando no hay prefix-cache real o no hay MTP/speculative. EXL3 muestra `Cached -` salvo `cached_pages>0`; `Drafted -` sin `draft_model=mtp`. vLLM muestra `Drafted` solo con `per_request_spec_decode_metrics` activo y modelo speculative. No hay fabricacion.
*   **`BrokenPipeError` en guard**: cliente cerro stream; no critico.
*   **Instalacion colgada compilando `llama.cpp`**: tarda 10-20 min con CUDA; usa `--llama-cpp-mode prebuilt` si el usuario quiere rapido.
*   **Permisos `system`**: el installer re-ejecuta con `sudo -E` automaticamente (`maybe_reexec_system_install`); no intentes `sudo` manual si el LLM no tiene `NOPASSWD`.

## 9. Checklist final para el LLM antes de dar por terminado

- [ ] `Q1-Q8` preguntadas o defaults justificados + comando `install` ejecutado con `--dry-run` primero
- [ ] `heimdall-gateway info` muestra `reachable` en API y UI
- [ ] `curl /v1/models` responde `200` (aunque sea lista vacia)
- [ ] Si el usuario pidio modelo, `heimdall-gateway list` lo muestra y `curl /v1/chat/completions` con ese `model` devuelve `choices`
- [ ] `heimdall-gateway config-keys --format json | jq '.logging.llama_swap'` muestra `{"logToStdout":"both","logLevel":"info"}` o valor configurado (whitelist both/proxy/upstream/none)
- [ ] `curl :11436/api/metrics/activity` y `jq '.timings'` verificados por backend (llama: slot print_timing + timings, EXL3: timings real Drafted/Prefill/Decode Cached - salvo real, vLLM: prompt/generated + spec-decode solo con flag)
- [ ] `config-migrate` segunda pasada idempotente verificada (`changed==False`)
- [ ] `heimdall-gateway logs --lines 200 --journal | grep timing` interpretado correctamente (EXL3/vLLM sin print_timing es esperado)
- [ ] Informado al usuario de `models-dir`, `mode`, `public-host:port`, y si `api_auth`/`api_https` estan activos

## 10. Referencias

*   `README.md` - overview, arquitectura, API, troubleshooting.
*   `llamacpp_stack/install.py` - fuente de verdad de prompts y flags (`prompt_bool`, `prompt_choice`, `resolve_*`, `build_cli_parser`).
*   `llamacpp_stack/cli.py: build_help_epilog`, `install --help`, `heimdall-gateway info --help`.
*   `docs/VLLM-BETA.md`, `docs/LOCAL_OLLAMA_SETUP.md`, `docs/arg-hyphen-conventions.md`.
*   `llamacpp_stack/bundle/llama_server_defaults.yaml` - defaults y presets por `gpu_count`.

## Heimdall Autotune Skill

`skills/heimdall-autotune/SKILL.md` - Loop automático para optimizar `qwen3.8` etc. tras `add` o `optimizar`. Valida con `parallel 2/3` y guarda snapshot en `configs/history/`.

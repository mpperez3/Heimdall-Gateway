Convención de guiones para argumentos de `llama-server`

Resumen
- Algunos flags relacionados con el comportamiento de "fit" (`fit`, `fitt`, `fitc`) usan la forma de guion único (por ejemplo `-fit`, `-fitt`, `-fitc`) porque el binario upstream espera la forma histórica de guion corto y no soporta la forma larga `--fit`/`--fitt`.
- Otros flags (p. ej. `--flash-attn`, `--batch-size`, etc.) usan la forma larga de doble guion. Mantener esta diferencia es intencional y necesaria.

Por qué
- El análisis del binario `llama-server` y la experiencia previa muestran que ciertas opciones históricas se parsean como opciones de guion corto; emitir `--fitt` provoca `invalid argument: --fitt` en tiempo de ejecución.

Qué se hizo
- `llamacpp_stack/cli.py` ahora emite `-fit` y `-fitt` (ya se había normalizado `fitc` a `-fitc`).
- Se añadió una prueba unitaria `tests/test_flag_prefixes.py` que verifica que las tres banderas (`fit`, `fitt`, `fitc`) se generen con guion único y no con doble guion.

Cómo evitar regresiones
- Si añades/ajustas mapeos en `_append_llama_server_flag`, actualiza o añade pruebas equivalentes en `tests/` para asegurar la forma correcta del prefijo.
- Antes de lanzar cambios que modifiquen `build_llama_server_command` o `_append_llama_server_flag`, ejecuta `pytest tests/test_flag_prefixes.py`.

Contacto
- Si el upstream cambia el parser (p. ej. añade forma larga), actualiza este documento y las pruebas para reflejar la nueva convención.

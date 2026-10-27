# Entrega 3: pruebas, validación de datos y seguridad (rama `feat/deepchecks`)

Aquí les cuento qué hice en mi parte de la entrega 3 (validación de datos con DeepChecks), por qué lo hice así y qué nos falta, con pasos concretos para cada pendiente.

## En corto

```bash
pip install -r requirements.txt   # obligatorio: cambiaron versiones (abajo explico)
make test                          # 51 pruebas, no necesita datos
make security                      # bandit y pip-audit
make validate                      # DeepChecks sobre P1, P2 y P3 (necesita make ingest)
```

| Lo que pide la entrega | Cómo va |
|---|---|
| Validación de datos con DeepChecks (integridad, nulos, duplicados, drift) | Listo |
| Evidencia de DeepChecks en el README con capturas | Listo |
| Pruebas unitarias de casos normales y de borde | Listo (`tests/test_metrics.py`) |
| Prueba de reproducibilidad | Listo (`tests/test_reproducibility.py`) |
| Seguridad: `.gitignore`, variables de entorno, bandit o pip-audit con evidencia | Listo (sección [Seguridad](../README.md#seguridad) del README) |
| Pruebas con un solo comando | Listo (`make test`) |
| CI con GitHub Actions | Listo: lint, pruebas y `make security` en cada PR |
| Commits de todos | **Falta**: cada quien con su parte |

## Qué hice y por qué

Un commit por tema:

1. **`fix(plots)`**: la curva PR usaba `curve_kwargs`, que solo existe desde scikit-learn 1.7. Ahora usa `color=`, que funciona en todas las versiones. Se ve igual.
2. **`build(deps)`: versiones compatibles con DeepChecks.** Este es el cambio más delicado, así que lo explico con calma abajo.
3. **`feat(validation)`: `mibici/validation.py` y `make validate`.**
   - Por pregunta corren dos suites: **Integridad** (nulos, tipos mezclados, duplicados, columnas constantes y etiquetas en conflicto, sobre train y sobre test) y **Train contra test** (drift de features y de la etiqueta, categorías nuevas y tamaño de test).
   - Valida las mismas columnas que usa cada modelo (`P1_FEATURES`, `P2_FEATURES`, `p3_feature_columns`), así que si alguien cambia las features, la validación se entera sola.
   - P1 y P2 tienen millones de filas, así que se revisa una muestra fija de 50,000 por conjunto (`VALIDATION_SAMPLE_SIZE` en `config.py`, con `SEED`). Tarda unos 15 segundos en total.
   - Si falla la integridad, el comando termina con error. El drift solo avisa: la partición es temporal y un cambio entre 2024-2025 y 2026 es esperado. Lo documentamos en vez de bloquear.
   - Guarda un HTML por suite en `reports/deepchecks/`, que agregué al `.gitignore` porque son archivos generados de hasta 34 MB.
4. **`test(validation)`: 6 pruebas con tablas sintéticas.** Comprueban que una tabla limpia pasa y que las suites detectan nulos, duplicados, una columna constante y una feature desplazada. No usan los datos de MiBici, así que corren en CI.
5. **`ci`: workflow de GitHub Actions** (`.github/workflows/ci.yml`). En cada PR y en cada push a `main` instala `requirements.txt` y corre `make lint` y `make test`. DeepChecks entra con las pruebas sintéticas; `make validate` no, porque necesita los datos de MiBici y no se versionan.
6. **`docs`**: sección [Validación de datos](../README.md#validación-de-datos) en el README con capturas, [`validacion_datos.md`](validacion_datos.md) con los hallazgos y este documento.

### Lo que encontró DeepChecks

Detalle en [`validacion_datos.md`](validacion_datos.md). Lo importante:

- **Integridad:** sin nulos ni tipos mezclados. Menos de 1 % de duplicados en P1 y P2: son estaciones sin actividad, no errores.
- **Columnas constantes en P3:** en train no hay ninguna salida entre las 2:00 y las 4:59, así que `dep_wd_h02` a `h04` y `dep_we_h02` a `h04` valen 0 en todas las estaciones. Las declaré como excepción conocida (`KNOWN_CONSTANT_COLUMNS`), así `make validate` solo falla si aparece una nueva. **Para quien lleve P3:** conviene evaluar quitarlas de las features.
- **Drift:** en P1 y P2 solo `month`, porque train tiene los 12 meses y test solo mayo a agosto. En P3 cambia la proporción de salidas de fin de semana en algunas horas (5:00, 16:00 a 18:00).

### Por qué cambiaron las versiones

DeepChecks 0.19.1 es la última versión y no funciona con lo que teníamos:

| Paquete | Antes | Ahora | Por qué |
|---|---|---|---|
| numpy | 2.5.3 | 1.26.4 | DeepChecks usa `np.Inf`, que numpy 2 quitó |
| scikit-learn | 1.9.1 | 1.5.2 | DeepChecks usa el scorer `max_error`, que se quitó en 1.6 (probé 1.6, 1.7 y 1.8 y fallan) |
| pandas | 3.0.5 | 2.2.3 | scikit-learn 1.5.2 salió antes que pandas 3 y no lo soporta; con 2.2.3 probé todo de punta a punta |
| scipy | sin fijar | 1.14.1 | la 1.17 usa `np.long`, que no existe en numpy 1.26 |
| category_encoders, statsmodels | no estaban | 2.6.4, 0.14.4 | las usa DeepChecks; sus versiones nuevas piden numpy 2 o scikit-learn 1.6 o posterior |
| setuptools | sin fijar | 80.9.0 | desde la 81 ya no trae `pkg_resources`, que DeepChecks importa |

Preferí bajar versiones en lugar de tener un segundo entorno solo para DeepChecks: así sigue habiendo un solo `requirements.txt` y todo es replicable con un `pip install`. Fijé a mano las dependencias de DeepChecks porque el resolvedor elegía versiones que se instalaban pero tronaban al importar.

**¿Cambian los modelos?** Entrené los 18 modelos con los hiperparámetros por defecto en los dos entornos. 14 dan exactamente lo mismo. Los otros 4 cambian en la tercera o cuarta cifra decimal y ningún champion ni challenger cambia de lugar:

| Modelo | Métrica | Antes (sklearn 1.9.1) | Ahora (sklearn 1.5.2) |
|---|---|---|---|
| P1 HGB | `valid_wape` | 0.1767 | 0.1760 |
| P1 HGB | `test_mae` | 5.2998 | 5.2714 |
| P2 HGB | `valid_pr_auc` | 0.4826 | 0.4817 |
| P2 random forest | `valid_pr_auc` | 0.4516 | 0.4519 |
| P3 MiniBatch KMeans | `train_profile_silhouette` | 0.0864 | 0.0862 |
| P3 MiniBatch KMeans | `test_stability_ari` | 0.4697 | 0.4492 |

Los modelos registrados en el `mlflow.db` de cada quien se guardaron con scikit-learn 1.9 y quizá no carguen con 1.5. Si les pasa, borren `mlflow.db` y `mlartifacts/` y vuelvan a correr `make all`. Las cifras del README y del reporte de la entrega 2 son las de la corrida oficial y se quedan como están.

## Qué falta y cómo hacerlo

### 1. Pruebas de modelado y reproducibilidad

Funciones sin prueba que conviene cubrir, con casos normales y de borde:

- `best_f1_threshold` y `ThresholdClassifier` (`mibici/modeling/p2.py`): con un `y_score` que separa perfecto, el umbral debe dar F1 = 1; con todo de una sola clase, no debe tronar.
- `regression_metrics` (`p1.py`), `classification_metrics` (`p2.py`) y `cluster_metrics` (`p3.py`): resultados conocidos con 4 o 5 valores a mano (por ejemplo, predicción perfecta da MAE 0 y WAPE 0).
- `block_start` (`features.py`): un viaje a las 23:59 cae en el bloque de las 21:00 y uno a las 5:59 queda fuera (los bloques empiezan a las 6:00).
- `compute_data_version` (`dataset.py`): mismo manifiesto da la misma versión; si cambia un hash, cambia la versión.

**Prueba de reproducibilidad:** entrenar el mismo modelo dos veces con `SEED` sobre datos sintéticos y comparar las predicciones. Algo así:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from mibici.config import SEED

def test_training_is_reproducible():
    X, y = make_synthetic_data()  # DataFrame pequeño con rng = np.random.default_rng(0)
    first = HistGradientBoostingRegressor(random_state=SEED).fit(X, y).predict(X)
    second = HistGradientBoostingRegressor(random_state=SEED).fit(X, y).predict(X)
    np.testing.assert_array_equal(first, second)
```

Lo ideal es llamar a las funciones de verdad (`p1_hgb(parts, hp)`) con un `parts` sintético de train, valid y test, para probar nuestro código y no solo scikit-learn.

### 2. Seguridad

- **pip-audit:** ya lo corrí (`pip install pip-audit` y luego `pip-audit` con el entorno activo). Encontró 5 vulnerabilidades en 2 paquetes; ninguna nos afecta hoy, pero hay que documentarlas en el README:
  - **mlflow 3.16.0** (CVE-2026-53995 y CVE-2026-115886): la primera es del backend FileStore (usamos SQLite) y la segunda requiere tener `trulens` instalado (no lo tenemos). Se arreglan con **mlflow 3.17.2**; comprobé que se instala sin conflictos con las versiones nuevas. Quien lo suba debe correr `make all` para confirmar que todo sigue funcionando.
  - **setuptools 80.9.0** (PYSEC-2026-3447 y GHSA-xmj3-9gf7-h52w): son de construir sdists con `MANIFEST.in` y de instalar eggs en Windows; no hacemos ninguna de las dos cosas. No se puede subir mientras DeepChecks necesite `pkg_resources`, así que se documenta como riesgo aceptado.
- **bandit:** `pip install bandit` y `bandit -r mibici`. Revisen cada hallazgo: si es falso positivo, se marca con `# nosec` y una explicación corta; si no, se corrige. Guarden la salida como evidencia.
- **`.env.example`:** `config.py` ya carga `.env` con `load_dotenv()` y `.env` está en `.gitignore`. Falta un `.env.example` versionado con las variables que se pueden configurar, sin valores secretos (hoy solo `MLFLOW_TRACKING_URI=http://127.0.0.1:5001`).
- **Revisar el historial:** `git log -p | grep -iE "password|token|secret|api_key"` para confirmar que nunca se subió una credencial.
- En el README, una sección de seguridad con esto y las capturas de bandit y pip-audit.

### 3. CI con GitHub Actions

El workflow ya existe (`.github/workflows/ci.yml`) y corre `make lint` y `make test`. Falta que quien haga seguridad le agregue bandit y pip-audit, **después** de resolver las vulnerabilidades de la sección anterior; si se agregan antes, todos los PR quedan en rojo.

1. Agregar `bandit` y `pip-audit` con versión fija en `requirements.txt`, debajo de `# Calidad`.
2. Subir mlflow a 3.17.2 y comprobar con `make all` que todo sigue funcionando.
3. Agregar al final de `ci.yml`:

```yaml
      - run: bandit -r mibici
      # setuptools no se puede subir mientras DeepChecks necesite pkg_resources (ver README).
      - run: pip-audit --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w
```

Cada vulnerabilidad que se ignore debe quedar explicada en el README.

### 4. README y commits

- Cada quien agrega su sección (pruebas, seguridad) con el comando para correrla y una captura. Para el CI sirve una captura de la pestaña *Checks* de un PR en verde.
- La entrega pide commits de todos: cada quien hace su parte en su propia rama y abre su PR.

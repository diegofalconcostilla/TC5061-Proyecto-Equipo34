# Auditoría de seguridad de dependencias

## 1. Objetivo

Evaluar la seguridad de las dependencias del proyecto MiBici mediante la herramienta `pip-audit`, identificando vulnerabilidades conocidas y posibles acciones de remediación.

## 2. Herramienta utilizada

Se utilizó `pip-audit` versión 2.10.1 para realizar dos análisis:

- Auditoría del entorno virtual instalado.
- Auditoría de las dependencias declaradas en `requirements.txt`.

## 3. Resultados

### 3.1. Entorno virtual

Se identificaron 10 registros de vulnerabilidad en tres paquetes:

| Paquete | Versión detectada | Versión correctiva |
|---|---|---|
| mlflow | 3.16.0 | 3.17.2 |
| pyasn1 | 0.6.4 | 0.6.5 |
| tornado | 6.5.10 | 6.6 |

### 3.2. Dependencias declaradas

La auditoría de `requirements.txt` identificó tres registros asociados a `setuptools`:

| Paquete | Versión detectada | Identificador | Versión correctiva |
|---|---|---|---|
| setuptools | 80.9.0 | PYSEC-2026-3447 | 83.0.0 |
| setuptools | 80.9.0 | PYSEC-2026-3447 | 83.0.0 |
| setuptools | 80.9.0 | GHSA-xmj3-9gf7-h52w | 84.1.0 |

### 3.3. Dependencia local

En ambos análisis, `mibici` aparece como dependencia no auditada porque no se encuentra publicada en PyPI. Esto no constituye por sí mismo una vulnerabilidad.

## 4. Recomendaciones

1. Revisar y actualizar las dependencias vulnerables a versiones corregidas.
2. Verificar la compatibilidad de las actualizaciones con MLflow y los experimentos existentes.
3. Sincronizar el entorno virtual con las versiones declaradas en `requirements.txt`.
4. Incorporar la auditoría de dependencias como parte del proceso de validación del proyecto.
5. Repetir el análisis después de aplicar las actualizaciones.

## 5. Conclusión

La auditoría permitió identificar dependencias con vulnerabilidades de seguridad conocidas tanto en el entorno de desarrollo como en las dependencias declaradas del proyecto.

Este análisis aporta una primera capa de gobernanza técnica y permite establecer acciones preventivas para mejorar la seguridad y confiabilidad del sistema MiBici.

**Herramienta:** pip-audit 2.10.1.

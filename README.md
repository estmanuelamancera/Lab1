### INFORME DE LABORATORIO #1.
### ANALÍSIS ESTADÍSTICOS DE LA SEÑAL
### DESCRIPCIÓN 
En este repositorio se llevó a cabo el análisis de señales biomédicas empleando tanto datos obtenidos de bases de datos como señales adquiridas de manera experimental. Inicialmente, se descargó una señal fisiológica, la cual fue importada a Python para su visualización y para el cálculo de parámetros estadísticos relevantes.

Adicionalmente, se capturó una señal mediante un osciloscopio, utilizando un sistema DAQ para su adquisición, y posteriormente se compararon sus características con las de la señal previamente analizada. Finalmente, se evaluó la influencia del ruido en ambas señales a través del cálculo de la Relación Señal-Ruido (SNR), examinando cómo este afecta la calidad y el comportamiento de la señal.
## OBJETIVOS 
1. Seleccionar, importar y examinar señales fisiológicas con el fin de realizar su procesamiento y análisis estadístico en Python.
2. Emplear funciones predefinidas en Python junto con procedimientos manuales para comparar las características estadísticas de señales simuladas frente a señales reales.
3. Generar una señal fisiológica en el laboratorio, adquirirla mediante un sistema DAQ y evaluar sus propiedades estadísticas principales.
### PROCESAMIENTO 
## PARTE A 
## PARTE B 
En la Parte B del laboratorio se generó experimentalmente una señal fisiológica mediante el generador de señales biológicas y posteriormente se adquirió utilizando un sistema DAQ conectado al computador a través de un puerto USB y configurado con el controlador NI-DAQmx. El dispositivo recibió la señal analógica, la convirtió a formato digital mediante su conversor analógico-digital (ADC) y la almacenó en un archivo con extensión `.csv`, que contenía las columnas correspondientes al tiempo de muestreo y a los valores de amplitud. La señal fue importada en Python mediante el entorno Spyder, donde se verificó su integridad, se graficó en el dominio del tiempo y se construyó su histograma para analizar la distribución de amplitudes.

Posteriormente, se calcularon estadísticos descriptivos como la media, la mediana, la varianza muestral, la desviación estándar, el coeficiente de variación, la asimetría, la curtosis y los valores máximo y mínimo, con el fin de caracterizar globalmente la señal adquirida. Finalmente, estos resultados se compararon con los obtenidos a partir de la señal descargada de PhysioNet, permitiendo identificar similitudes estructurales y pequeñas diferencias atribuibles al ruido del sistema de adquisición, a la cuantización del ADC y a posibles interferencias del entorno experimental.


# PROCESAMIENTO 


## PARTE C
En esta seccion se tomo la señal del apartado anterior (La señal obtenida del generador de señales) para posteriormente agregarle diferentes tipos de ruido y finalmnete compararlas entre si.

### PROCEDIMIENTO
Para aplicar diferentes ruidos a una señal primero debemos partir de la base teórica de la relación entre una señal y el ruido (SNR), por ello entraremos a revisar primero conceptos para posteriormente aplicar los siguientes tipos de ruidos:
1.Ruido gaussiano
2.Ruido impulso
3.Ruido tipo artefacto

### ¿QUÉ ES SNR?
Cuando se habla del SNR, se refiere a la relación señal ruido, la cual en términos simples mide la intensidad de nuestra señal en relación con una interferencia no deseada que en este caso es un tipo de ruido el cual se puede definir como culquier alteración que cambie la calidad de nuestra señal. Usamos el SNR en distintos campos de la ingeniería para cuantificar la claridad de las señales, esta se suele representar como un valor numérico en decibelios (db) utilizando una escala logarítmica.

Para calcular el SNR se utiliza la siguiente expresión: 

$$
SNR = \frac{P señal}{Pruido}
$$
### FUNCIÓN PARA CALCULAR EL SNR
```python

def calcular_snr(signal, noise):
    potencia_senal = np.mean(signal**2)
    potencia_ruido = np.mean(noise**2)
    return 10 * np.log10(potencia_senal / potencia_ruido)

```
## CÓDIGO Y GRÁFICA

### RUIDO GAUSSIANO
![GRÁFICA GAUSSIANO](https://github.com/estmanuelamancera/Lab1/blob/main/IMAGENES/ruido%20gauss.png?raw=true
)
#### CÓDIGO 

```python

sigma = 2 * np.std(senal)  # intensidad ajustable
ruido_gauss = np.random.normal(0, sigma, len(senal))

senal_gauss = senal + ruido_gauss
snr_gauss = calcular_snr(senal, ruido_gauss)

```
## RUIDO DE IMPULSO
![GRÁFICA IMPULSO](https://github.com/estmanuelamancera/Lab1/blob/main/IMAGENES/ruido%20impulso.png?raw=true)

#### CÓDIGO 
```python

senal_impulso = senal.copy()
ruido_impulso = np.zeros_like(senal)

prob = 0.05  # 1% de muestras afectadas
num_impulsos = int(prob * len(senal))

indices = np.random.choice(len(senal), num_impulsos, replace=False)
amplitud = 3 * np.std(senal)

for i in indices:
    valor = amplitud * np.random.choice([-1, 1])
    senal_impulso[i] += valor
    ruido_impulso[i] = valor

snr_impulso = calcular_snr(senal, ruido_impulso)


```


## RUIDO DE ARTEFACTO
![GRÁFICA ARTEFACTO](https://github.com/estmanuelamancera/Lab1/blob/main/IMAGENES/ruido%20artefacto.png?raw=true)

#### CÓDIGO 
```python
baseline = 0.5 * np.sin(2 * np.pi * 0.5 * t)      # 0.5 Hz
interferencia = 0.2 * np.sin(2 * np.pi * 60 * t) # 60 Hz

ruido_artefacto = baseline + interferencia
senal_artefacto = senal + ruido_artefacto

snr_artefacto = calcular_snr(senal, ruido_artefacto)


```

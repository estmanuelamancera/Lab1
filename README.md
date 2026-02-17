### INFORME DE LABORATORIO #1.


### PARTE C
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

### RUIDO GAUSSIANO
## CÓDIGO Y GRÁFICA

```python

sigma = 2 * np.std(senal)  # intensidad ajustable
ruido_gauss = np.random.normal(0, sigma, len(senal))

senal_gauss = senal + ruido_gauss
snr_gauss = calcular_snr(senal, ruido_gauss)

```


![GRÁFICA GAUSSIANO](https://github.com/estmanuelamancera/Lab1/blob/main/IMAGENES/ruido%20gauss.png?raw=true
)
## RUIDO DE IMPULSO
![GRÁFICA IMPULSO](https://github.com/estmanuelamancera/Lab1/blob/main/IMAGENES/ruido%20impulso.png?raw=true)

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

```python
baseline = 0.5 * np.sin(2 * np.pi * 0.5 * t)      # 0.5 Hz
interferencia = 0.2 * np.sin(2 * np.pi * 60 * t) # 60 Hz

ruido_artefacto = baseline + interferencia
senal_artefacto = senal + ruido_artefacto

snr_artefacto = calcular_snr(senal, ruido_artefacto)


```

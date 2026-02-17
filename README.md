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

```


![GRÁFICA GAUSSIANO](https://github.com/estmanuelamancera/Lab1/blob/main/IMAGENES/ruido%20gauss.png?raw=true
)
## RUIDO DE IMPULSO
![GRÁFICA IMPULSO](https://github.com/estmanuelamancera/Lab1/blob/main/IMAGENES/ruido%20impulso.png?raw=true)
## RUIDO DE ARTEFACTO
![GRÁFICA ARTEFACTO](https://github.com/estmanuelamancera/Lab1/blob/main/IMAGENES/ruido%20artefacto.png?raw=true)

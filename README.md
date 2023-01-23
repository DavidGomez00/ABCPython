# Sobre el repositorio
Este repositorio ha sido creado a partir del código encontrado en https://github.com/abcolony/ABCPython, donde Omur ha implementado el algoritmo Artificial Bee Colony Optimization (ABC) en el leguaje Python.

# Artificial Bee Colony
El algortimo de Artificial Bee Colony fue introducido en el trabajo de Karaboga et al. «An idea based on honeybee swarm for numerical optimization» en 2005. Se trata de un algortimo de Swarm Intelligence inspirado en el comportamiento de las abejas.

## ABCPython
Bajo la carpeta ABCPython se encuentra una imeplementación del algoritmo ABC, ligeramente modificada del repositorio https://github.com/abcolony/ABCPython. Se pueden modificar los parámetros del algorirmo en el ficehro ABC.ini y puede ejecutarse mediante el archivo ABCAlgorithm.py.

## BDLDABCPython
Bajo la carptera BDLDABCPython se encuentra una implementación del algoritmo BDLDABC. Este algortimo es una variante del algortimo ABC que combate la falta de capacidad explotativa del algoritmo ABC mediante el uso de distintas ecuaciones de búsqueda en las distintas fases del algoritmo. Este método se presentó en el trabajo de Wang et al. Y. «A labor division artificial bee colony algorithm based on behavioral development». Para mayor entendimiento del algoritmo ABC y la variante BDLDABC consultar el artículo original, disponible en: https://www.sciencedirect.com/science/article/abs/pii/S0020025522004972   DOI: 10.1016/j.ins.2022.05.065

Se pueden modificar los parámetros del algorirmo en el ficehro ABC.ini y puede ejecutarse mediante el archivo ABCAlgorithm.py.

## MFABC
Bajo la carpeta MFABCpython se encuentra una implementación del algortimo presentado por Song et al. en «A multi-strategy fusion artificial bee colony algorithm with small population». En este trabajo los autores modifican el algortmo ABC para, al igual que en el caso de BDLDABC, mejorar la capacidad de explotación del algoritmo. El algortimo de Mult-strategy Function Artificial Bee Colony utiliza una aproximación en la que se combinan la modificación de las ecuaciones de búsqueda con la combinación de varias estrategias de búsqueda. Para comprender mejor el funcionamiento del algortimo MFABC, consultar el artículo original, disponible en https://www.sciencedirect.com/science/article/abs/pii/S0957417419306396    DOI: 10.1016/j.eswa.2019.112921

Se pueden modificar los parámetros del algorirmo en el ficehro ABC.ini y puede ejecutarse mediante el archivo ABCAlgorithm.py.

## License
[MIT](https://choosealicense.com/licenses/mit/)

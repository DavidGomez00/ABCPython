__author__ = "Omur Sahin"

import sys
import numpy as np
from deap.benchmarks import *
from scipy.spatial.distance import cdist
import progressbar

class ABC:

    def __init__(_self, conf):
        _self.conf = conf
        _self.foods = np.zeros((_self.conf.FOOD_NUMBER, _self.conf.DIMENSION))
        _self.roles = np.zeros(_self.conf.FOOD_NUMBER)
        _self.f = np.ones((_self.conf.FOOD_NUMBER))
        _self.fitness = np.ones((_self.conf.FOOD_NUMBER)) * np.iinfo(int).max
        _self.degrees = np.zeros(_self.conf.FOOD_NUMBER)
        _self.trial = np.zeros((_self.conf.FOOD_NUMBER))
        _self.role_trial = np.zeros((_self.conf.FOOD_NUMBER))
        _self.prob = [0 for x in range(_self.conf.FOOD_NUMBER)]
        _self.solution = np.zeros((_self.conf.DIMENSION))
        _self.globalParams = [0 for x in range(_self.conf.DIMENSION)]
        _self.globalTime = 0
        _self.evalCount = 0
        _self.cycle = 0
        _self.experimentID = 0
        _self.globalOpts = list()
        _self.globalOptDegree = 0
        _self.LIMIT = (int)(_self.conf.BETA * _self.conf.RLIMIT)

        if (_self.conf.SHOW_PROGRESS):
            _self.progressbar = progressbar.ProgressBar(max_value=_self.conf.MAXIMUM_EVALUATION)
        if (not(conf.RANDOM_SEED)):
            random.seed(conf.SEED)

    def calculate_function(_self, sol):
        try:
            if (_self.conf.SHOW_PROGRESS):
                _self.progressbar.update(_self.evalCount)
            return _self.conf.OBJECTIVE_FUNCTION(sol)

        except ValueError as err:
            print(
                "An exception occured: Upper and Lower Bounds might be wrong. (" + str(err) + " in calculate_function)")
            sys.exit()

    def calculate_fitness(_self, fun):
        _self.increase_eval()
        if fun >= 0:
            result = 1 / (fun + 1)
        else:
            result = 1 + abs(fun)
        return result

    def increase_eval(_self):
        _self.evalCount += 1

    def stopping_condition(_self):
        '''TODO: Parar cuando se sencuentre el 0?'''
        status = bool(_self.evalCount >= _self.conf.MAXIMUM_EVALUATION)
        if(_self.conf.SHOW_PROGRESS):
          if(status == True and not( _self.progressbar._finished )):
               _self.progressbar.finish()
        return status

    def memorize_best_source(_self):
        for i in range(_self.conf.FOOD_NUMBER):
            if (_self.f[i] < _self.globalOpt and _self.conf.MINIMIZE == True) or (_self.f[i] >= _self.globalOpt and _self.conf.MINIMIZE == False):
                _self.globalOpt = np.copy(_self.f[i])
                _self.globalParams = np.copy(_self.foods[i][:])

    def init(_self, index):
        if (not (_self.stopping_condition())):
            for i in range(_self.conf.DIMENSION):
                _self.foods[index][i] = random.random() * (_self.conf.UPPER_BOUND - _self.conf.LOWER_BOUND) + _self.conf.LOWER_BOUND
            _self.solution = np.copy(_self.foods[index][:])
            _self.f[index] = _self.calculate_function(_self.solution)[0]
            _self.fitness[index] = _self.calculate_fitness(_self.f[index])
            _self.trial[index] = 0
            _self.roles[index] = 0
            _self.role_trial[index] = 0

    def initial(_self):
        for i in range(_self.conf.FOOD_NUMBER):
            _self.init(i)
        _self.globalOpt = np.copy(_self.f[0])
        _self.globalParams = np.copy(_self.foods[0][:])

    def start(_self):
        i = 0
        while (not (_self.stopping_condition())):
            if (_self.roles[i] == 0):
                '''Employed bees'''
                # Se escoge un parámetro aleatorio
                r = random.random()
                _self.param2change = (int)(r * _self.conf.DIMENSION)

                # Se escoge una solución aleatoria
                r = random.random()
                _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)

                # Se escoge otro distinto a i
                while _self.neighbour == i:
                    r = random.random()
                    _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)

                # Se guarda la información de la solución i
                _self.solution = np.copy(_self.foods[i][:])

                # Se genera la nueva solución
                phi = random.random()
                _self.solution[_self.param2change] = _self.globalParams[_self.param2change] + \
                    (_self.globalParams[_self.param2change] - _self.foods[_self.neighbour][_self.param2change]) * (phi - 0.5) * 2

                # Control de rangos en los valores cambiados.
                if _self.solution[_self.param2change] < _self.conf.LOWER_BOUND:
                    _self.solution[_self.param2change] = _self.conf.LOWER_BOUND
                if _self.solution[_self.param2change] > _self.conf.UPPER_BOUND:
                    _self.solution[_self.param2change] = _self.conf.UPPER_BOUND
                
                # Se calculan la función y el fitness para la nueva solución
                _self.ObjValSol = _self.calculate_function(_self.solution)[0]
                _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)

                # Se aumenta el contador de búsquedas en este rol
                _self.role_trial[i] = _self.role_trial[i] + 1 
            
                if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                    # Se calcula el grado de mejora
                    _self.degrees[i] = 1 - ( _self.f[i]/ _self.ObjValSol)
                    
                    # En caso de mejorar el fitness se actualiza la solución
                    _self.trial[i] = 0  # resetea el trial
                    _self.foods[i][:] = np.copy(_self.solution) # sustitye la solución
                    _self.f[i] = _self.ObjValSol # valor de la función objetivo
                    _self.fitness[i] = _self.FitnessSol # Valor del fitness para esta solución
                    
                    # Se comprueba si el grado de mejora es suficiente para ignorar los intentos por rol
                    if ((_self.degrees[i] < _self.conf.MIN_IMPROVEMENT) and (_self.role_trial[i] > _self.conf.RLIMIT)):
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 1 # Pasa a ser onlooker bee

                else:
                    # Si no hay mejora se aumenta el trial.
                    _self.degrees[i] = 0
                    _self.trial[i] = _self.trial[i] + 1
                    # Se comprueba si debe desarrollarse a la siguiente fase
                    if ((_self.trial[i] > _self.LIMIT) or (_self.role_trial[i] > _self.conf.RLIMIT)):
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 1 # Pasa a ser onlooker bee
                
            elif (_self.roles [i] == 1):
                '''Onlooker bees'''
                # Se realiza el cálculo del mejor vecino
                distances = cdist(_self.foods, [_self.foods[i]], 'euclidean')
                k_nearest =  np.argsort(distances.flatten())[:_self.conf.K]

                best_neighbour = np.copy(_self.foods[k_nearest[np.argmax(_self.fitness.take(k_nearest, axis=0))]])

                # Se escoge el parámetro a cambiar
                r = random.random()
                _self.param2change = (int)(r * _self.conf.DIMENSION)
                # Se escoge un vecino aleatorio
                r = random.random()
                _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
                # Se escoge otro distinto a i
                while _self.neighbour == i:
                    r = random.random()
                    _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)

                # Se calcula la nueva solución
                _self.solution = np.copy(_self.foods[i][:])
                phi = random.random()
                psi = random.random()
                _self.solution[_self.param2change] = best_neighbour[_self.param2change] \
                            + (best_neighbour[_self.param2change] - _self.globalParams[_self.param2change]) * (phi - 0.5) * 2 \
                            + (best_neighbour[_self.param2change] - _self.foods[_self.neighbour][_self.param2change]) * (psi - 0.5) * 2
                            
                # Control de los rangos en los valores cambiados
                if _self.solution[_self.param2change] < _self.conf.LOWER_BOUND:
                    _self.solution[_self.param2change] = _self.conf.LOWER_BOUND
                if _self.solution[_self.param2change] > _self.conf.UPPER_BOUND:
                    _self.solution[_self.param2change] = _self.conf.UPPER_BOUND

                # Se calculan la función y el fitness para la nueva solución
                _self.ObjValSol = _self.calculate_function(_self.solution)[0]
                _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)

                # Se incrementa el contador de búsquedas en este rol
                _self.role_trial[i] = _self.role_trial[i] + 1

                if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                    # Se calcula el grado de mejora
                    _self.degrees[i] = 1 - ( _self.f[i]/ _self.ObjValSol)

                    # Se actualiza:
                    _self.trial[i] = 0  # resetea el trial
                    _self.foods[i][:] = np.copy(_self.solution) # sustitye la solución
                    _self.f[i] = _self.ObjValSol # valor de la función objetivo
                    _self.fitness[i] = _self.FitnessSol # Valor del fitness para esta solución

                    # Se comprueba si el grado de mejora es suficiente para ignorar los intentos por rol
                    if ((_self.degrees[i] < _self.conf.MIN_IMPROVEMENT) and (_self.role_trial[i] > _self.conf.RLIMIT)):
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 2 # Pasa a ser scout bee
                else:
                    # No hay mejora y se aumenta el trial
                    _self.degrees[i] = 0
                    _self.trial[i] = _self.trial[i] + 1
                    # Reversed development
                    if (_self.globalOptDegree > _self.conf.MI):
                        _self.foods[i] = np.copy(_self.globalParams)
                        _self.fitness[i] = _self.calculate_fitness(_self.globalOpt)
                        _self.f[i] = _self.globalOpt
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 0 # Pasa a ser employed bee
                    # Normal and accelerated development
                    elif ((_self.trial[i] > _self.LIMIT) or (_self.role_trial[i] > _self.conf.RLIMIT)):
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 2 # Pasa a ser scout bee
            
            else:
                '''Scout bees'''
                # Se escoge el parámetro a cambiar
                r = random.random()
                _self.param2change = (int)(r * _self.conf.DIMENSION)
                # Se escogen vecinos aleatorio
                r = random.random()
                _self.neighbour = (int)(r * _self.conf.FOOD_NUMBER)
                r = random.random()
                _self.neighbour1 = (int)(r * _self.conf.FOOD_NUMBER)
                r = random.random()
                _self.neighbour2 = (int)(r * _self.conf.FOOD_NUMBER)
                r = random.random()
                _self.neighbour3 = (int)(r * _self.conf.FOOD_NUMBER)

                # Se calcula la nueva solución
                phi = random.random()
                _self.solution = np.copy(_self.foods[i][:])        
                _self.solution[_self.param2change] = (
                    _self.foods[_self.neighbour][_self.param2change] + _self.foods[_self.neighbour1][_self.param2change]) / 2 \
                        + (_self.foods[_self.neighbour2][_self.param2change] - _self.foods[_self.neighbour3][_self.param2change]) * (phi -0.5) * 2
                
                # Control de rangos en los valores cambiados.
                if _self.solution[_self.param2change] < _self.conf.LOWER_BOUND:
                    _self.solution[_self.param2change] = _self.conf.LOWER_BOUND
                if _self.solution[_self.param2change] > _self.conf.UPPER_BOUND:
                    _self.solution[_self.param2change] = _self.conf.UPPER_BOUND

                # Se calculan la función y el fitness para la nueva solución
                _self.ObjValSol = _self.calculate_function(_self.solution)[0]
                _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)

                # Se aumenta el contador de búsquedas en este rol
                _self.role_trial[i] = _self.role_trial[i] + 1

                # Si mejora, se actualiza el grado de mejora
                if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                    _self.degrees[i] = 1 - (_self.ObjValSol / _self.f[i])
                    # Se resetea el número de búsquedas fallidas
                    _self.trial[i] = 0

                    # Se comprueba si el grado de mejora es suficiente para ignorar los intentos por rol
                    if ((_self.degrees[i] < _self.conf.MIN_IMPROVEMENT) and (_self.role_trial[i] > _self.conf.RLIMIT)):
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 2 # Pasa a ser scout bee
                        
                else:
                    # No hay mejora y se aumenta el trial.
                    _self.trial[i] = _self.trial[i] + 1
                    _self.degrees[i] = 0
                    
                    # Se realiza el cálculo del mejor vecino
                    distances = cdist(_self.foods, [_self.foods[i]], 'euclidean')
                    k_nearest =  np.argsort(distances.flatten())[:_self.conf.K]
                    # Se obtiene el grado de mejora del mejor vecino
                    index = k_nearest[np.argmax(_self.fitness.take(k_nearest, axis=0))]
                    bestNeighbourDegree = _self.degrees[index]
                    
                    # Reversed development - Global
                    if ((_self.globalOptDegree > _self.conf.MI) and (_self.globalOptDegree > bestNeighbourDegree)):
                        # Busca alrededor de la solución global
                        _self.foods[i] = np.copy(_self.globalParams)
                        _self.fitness[i] = _self.calculate_fitness(_self.globalOpt)
                        _self.f[i] = _self.globalOpt
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 0 # Pasa a ser employed bee
                    # Reversed development - Local
                    elif ((bestNeighbourDegree > _self.conf.MI) and (bestNeighbourDegree >= _self.globalOptDegree)):
                        # Busca alrededor de la mejor solución entre los vecinos
                        _self.foods[i] = np.copy(_self.foods[index])
                        _self.fitness[i] = np.copy(_self.fitness[index])
                        _self.f[i] = np.copy(_self.f[index])
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 0 # Pasa a ser employed bee
                    
                    # Normal and accelerated development.
                    elif ((_self.trial[i] > _self.LIMIT) or (_self.role_trial[i] > _self.conf.RLIMIT)):
                        _self.trial[i] = 0
                        _self.role_trial[i] = 0
                        _self.roles[i] = 0 # Pasa a ser employed bee

                # Se sustituye la solución
                _self.foods[i][:] = np.copy(_self.solution) # sustitye la solución
                _self.f[i] = _self.ObjValSol # valor de la función objetivo
                _self.fitness[i] = _self.FitnessSol # Valor del fitness para esta solución

            i += 1
            i = i % _self.conf.FOOD_NUMBER
            if (i==0):
                _self.increase_cycle()
                _self.memorize_best_source()
         
    def increase_cycle(_self):
        _self.globalOpts.append(_self.globalOpt)
        _self.cycle += 1
        
    def setExperimentID(_self,run,t):
        _self.experimentID = t+"-"+str(run)


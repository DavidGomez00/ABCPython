__author__ = "Omur Sahin"

import sys
import numpy as np
from deap.benchmarks import *
import progressbar

class ABC:

    def __init__(_self, conf):
        _self.conf = conf
        _self.foods = np.zeros((_self.conf.FOOD_NUMBER, _self.conf.DIMENSION))
        _self.f = np.ones((_self.conf.FOOD_NUMBER))
        _self.fitness = np.ones((_self.conf.FOOD_NUMBER)) * np.iinfo(int).max
        _self.trial = np.zeros((_self.conf.FOOD_NUMBER))
        _self.prob = [0 for x in range(_self.conf.FOOD_NUMBER)]
        _self.solution = np.zeros((_self.conf.DIMENSION))
        _self.globalParams = [0 for x in range(_self.conf.DIMENSION)]
        _self.globalTime = 0
        _self.globalIndex = 0
        _self.evalCount = 0
        _self.cycle = 0
        _self.pf = 0.5
        _self.experimentID = 0
        _self.evRatioPHABC = 0
        _self.evRatioABCbest = 0
        _self.avgEvolutionDegreePHABC = 0
        _self.avgEvolutionDegreeABCbest = 0
        _self.degrees = np.zeros(_self.conf.FOOD_NUMBER)
        _self.globalOpts = list()
        _self.eq = np.zeros(_self.conf.FOOD_NUMBER)

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
                _self.globalIndex = i

    def init(_self, index):
        if (not (_self.stopping_condition())):
            for i in range(_self.conf.DIMENSION):
                _self.foods[index][i] = random.random() * (_self.conf.UPPER_BOUND - _self.conf.LOWER_BOUND) + _self.conf.LOWER_BOUND
            _self.solution = np.copy(_self.foods[index][:])
            _self.f[index] = _self.calculate_function(_self.solution)[0]
            _self.fitness[index] = _self.calculate_fitness(_self.f[index])
            _self.trial[index] = 0

    def initial(_self):
        for i in range(_self.conf.FOOD_NUMBER):
            _self.init(i)
        _self.globalOpt = np.copy(_self.f[0])
        _self.globalParams = np.copy(_self.foods[0][:])

    def send_employed_bees(_self):
        i = 0
        while (i < _self.conf.FOOD_NUMBER) and (not (_self.stopping_condition())):
            # Se escoge un parámetro aleatorio
            r = random.random()
            _self.param2change = (int)(r * _self.conf.DIMENSION)

            # Se escogen 2 soluciones distintas aleatorias distintas que no sean la mejor solución ni la actual
            neighbours = np.random.choice(np.delete(np.arange(_self.foods.shape[0]), [_self.globalIndex, i]), 2, replace=False)
            n1, n2 = neighbours[0], neighbours[1]

            # Se guarda la información de la solución i
            _self.solution = np.copy(_self.foods[i][:])

            # Si el núnmero de ciclo es múltiplo de 4 se usa CABC
            if (_self.cycle % 4 == 0):
                # Se indica que la solución usa la ecuación CABC
                _self.eq[i] = 0
                # Se calcula la nueva solución
                _self.solution[_self.param2change] = (_self.foods[n1][_self.param2change]) \
                    + (_self.foods[n1][_self.param2change] - _self.foods[n2][_self.param2change]) \
                        * (random.uniform(-1, 1))
            # Se usa PHABC
            elif (random.random() < _self.pf):
                # Se indica que la solución usa la ecuación PHABC
                _self.eq[i] = 1
                # Se calcula la nueva solución
                _self.solution[_self.param2change] = (_self.foods[n1][_self.param2change]) \
                    + (_self.foods[n1][_self.param2change] - _self.foods[n2][_self.param2change]) \
                        * (random.uniform(-0.75, 0.75)) \
                            + (_self.globalParams[_self.param2change] - _self.foods[n1][_self.param2change]) \
                                * (random.uniform(0, 0.5))
            # Se usa ABC/best
            else:
                # Se indica que la solución usa la ecuación ABCbest
                _self.eq[i] = 2
                # Se calcula la nueva solución
                _self.solution[_self.param2change] = (_self.globalParams[_self.param2change]) \
                    + (_self.foods[n1][_self.param2change] - _self.foods[n2][_self.param2change]) \
                        * (random.uniform(-1, 1))

            # Control de rangos en los valores cambiados.
            if _self.solution[_self.param2change] < _self.conf.LOWER_BOUND:
                _self.solution[_self.param2change] = _self.conf.LOWER_BOUND
            if _self.solution[_self.param2change] > _self.conf.UPPER_BOUND:
                _self.solution[_self.param2change] = _self.conf.UPPER_BOUND
            
            # Se evalúa la nueva solución
            _self.ObjValSol = _self.calculate_function(_self.solution)[0] # Valor
            _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)   # Fitness
            
            if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                # Se calcula el grado de mejora
                _self.degrees[i] = _self.ObjValSol - _self.f[i]

                # Se actualiza la solución: 
                _self.trial[i] = 0  # resetea el trial
                _self.foods[i][:] = np.copy(_self.solution) # sustitye la solución
                _self.f[i] = _self.ObjValSol # valor de la función objetivo
                _self.fitness[i] = _self.FitnessSol # Valor del fitness para esta solución

                # Se actualiza el evolution ratio de las ecuaciones
                # PHABC
                if (_self.eq[i] == 1):
                    usosPHABC = np.where(_self.eq == 1)
                    _self.avgEvolutionDegreePHABC = np.sum(_self.degrees[usosPHABC]) / len(usosPHABC)
                    _self.evRatioPHABC = (_self.avgEvolutionDegreePHABC) / (_self.avgEvolutionDegreePHABC + _self.avgEvolutionDegreeABCbest)
                
                # ABCbest
                elif (_self.eq[i] == 2):
                    usosABCbest = np.where(_self.eq == 2)
                    _self.avgEvolutionDegreeABCbest = np.sum(_self.degrees[usosABCbest]) / len(usosABCbest)
                    _self.evRatioABCbest = (_self.avgEvolutionDegreeABCbest) / (_self.avgEvolutionDegreePHABC + _self.avgEvolutionDegreeABCbest)
                
            else:
                # Se actualiza el grado de mejora a 0
                _self.degrees[i] = 0
                # Aumenta el trial.
                _self.trial[i] = _self.trial[i] + 1 
            i += 1

    def update_pf(_self):
        '''Actualiza pf que es la probabilidad de
        escoger una ecuación de búsqueda u otra'''
        # Cómputo del nuevo pf
        if ((_self.evRatioPHABC + _self.evRatioABCbest) == 0):
            _self.pf = 0.1
        else:
            normEvRatioPHABC = _self.evRatioPHABC / (_self.evRatioPHABC + _self.evRatioABCbest)
            _self.pf = _self.pf * (1 - _self.conf.UR) + (normEvRatioPHABC * _self.conf.UR)

        # Control de valores mínimo/máximo
        if (_self.pf < 0.1): _self.pf = 0.1
        elif (_self.pf > 0.9): _self.pf = 0.9

    def send_onlooker_bees(_self):
        i = 0
        while (i < _self.conf.FOOD_NUMBER) and (not (_self.stopping_condition())):
            # Se escoge un parámetro aleatorio
            r = random.random()
            _self.param2change = (int)(r * _self.conf.DIMENSION)

            # Se escogen 2 soluciones distintas aleatorias distintas que no sean la mejor solución ni la actual
            neighbours = np.random.choice(np.delete(np.arange(_self.foods.shape[0]), [_self.globalIndex, i]), 2, replace=False)
            n1, n2 = neighbours[0], neighbours[1]

            # Se guarda la información de la solución i
            _self.solution = np.copy(_self.foods[i][:])

            if (random.random() < _self.pf):
                # Se indica que la solución usa la ecuación PHABC
                _self.eq[i] = 1
                # Se calcula la nueva solución
                _self.solution[_self.param2change] = (_self.foods[n1][_self.param2change]) \
                    + (_self.foods[n1][_self.param2change] - _self.foods[n2][_self.param2change]) \
                        * (random.uniform(-0.75, 0.75)) \
                            + (_self.globalParams[_self.param2change] - _self.foods[n1][_self.param2change]) \
                                * (random.uniform(0, 0.5))
            # Se usa ABC/best
            else:
                # Se indica que la solución usa la ecuación ABCbest
                _self.eq[i] = 2
                # Se calcula la nueva solución
                _self.solution[_self.param2change] = (_self.globalParams[_self.param2change]) \
                    + (_self.foods[n1][_self.param2change] - _self.foods[n2][_self.param2change]) \
                        * (random.uniform(-1, 1))
            
            # Control de rangos en los valores cambiados.
            if _self.solution[_self.param2change] < _self.conf.LOWER_BOUND:
                _self.solution[_self.param2change] = _self.conf.LOWER_BOUND
            if _self.solution[_self.param2change] > _self.conf.UPPER_BOUND:
                _self.solution[_self.param2change] = _self.conf.UPPER_BOUND
            
            # Se evalúa la nueva solución
            _self.ObjValSol = _self.calculate_function(_self.solution)[0] # Valor
            _self.FitnessSol = _self.calculate_fitness(_self.ObjValSol)   # Fitness
            
            if (_self.FitnessSol > _self.fitness[i] and _self.conf.MINIMIZE == True) or (_self.FitnessSol <= _self.fitness[i] and _self.conf.MINIMIZE == False):
                # Se calcula el grado de mejora
                _self.degrees[i] = _self.ObjValSol - _self.f[i]

                # Se actualiza la solución: 
                _self.trial[i] = 0  # resetea el trial
                _self.foods[i][:] = np.copy(_self.solution) # sustitye la solución
                _self.f[i] = _self.ObjValSol # valor de la función objetivo
                _self.fitness[i] = _self.FitnessSol # Valor del fitness para esta solución

                # Se actualiza el evolution ratio de las ecuaciones
                # PHABC
                usosPHABC = np.where(_self.eq == 1)
                _self.avgEvolutionDegreePHABC = np.sum(_self.degrees[usosPHABC]) / len(usosPHABC)
                # ABCbest
                usosABCbest = np.where(_self.eq == 2)
                _self.avgEvolutionDegreeABCbest = np.sum(_self.degrees[usosABCbest]) / len(usosABCbest)

                # Se calcula el ev. ratio
                _self.evRatioPHABC = (_self.avgEvolutionDegreePHABC) / (_self.avgEvolutionDegreePHABC + _self.avgEvolutionDegreeABCbest)
                _self.evRatioABCbest = (_self.avgEvolutionDegreeABCbest) / (_self.avgEvolutionDegreePHABC + _self.avgEvolutionDegreeABCbest)
            else:
                # Se actualiza el grado de mejora a 0
                _self.degrees[i] = 0
                # Aumenta el trial.
                _self.trial[i] = _self.trial[i] + 1 
            
            i += 1

    def send_scout_bees(_self):
        if (np.amax(_self.trial) > _self.conf.LIMIT):
            _self.init(_self.trial.argmax(axis = 0))
        
    def increase_cycle(_self):
        _self.globalOpts.append(_self.globalOpt)
        _self.cycle += 1
        
    def setExperimentID(_self,run,t):
        _self.experimentID = t+"-"+str(run)

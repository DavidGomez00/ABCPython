import os

import numpy as np
import pandas as pd

from decimal import Decimal

class Reporter:
    def __init__(_self, abcList):
        _self.abcList = abcList
        if(abcList[0].conf.PRINT_PARAMETERS):
            _self.print_parameters()

        if (abcList[0].conf.RUN_INFO):
            _self.run_info()
        if (abcList[0].conf.SAVE_RESULTS):
             _self.save_results()
        if (abcList[0].conf.RUN_INFO_COMMANDLINE):
            _self.command_line_print()


    def print_parameters(_self):
        for i in range(_self.abcList[0].conf.RUN_TIME):
            print(_self.abcList[i].experimentID,". run")
            for j in range(_self.abcList[0].conf.DIMENSION):
                print("Global Param[", j + 1, "] ", _self.abcList[i].globalParams[j])

    def run_info(_self):
        sum = []
        for i in range(_self.abcList[0].conf.RUN_TIME):
            print(_self.abcList[i].experimentID + " run: ", _self.abcList[i].globalOpt, " Cycle: ",
                  _self.abcList[i].cycle, " Time: ",
                  _self.abcList[i].globalTime)
            sum.append(_self.abcList[i].globalOpt)
        print("Mean: ",np.mean(sum)," Std: ",np.std(sum)," Median: ",np.median(sum))
        
    def command_line_print(_self):
        sum = []
        for i in range(_self.abcList[0].conf.RUN_TIME):
            sum.append(_self.abcList[i].globalOpt)
        print('%1.5E' % Decimal(np.mean(sum)))

    def save_results(_self):
        # Crea la carpeta de resultados
        if not os.path.exists(_self.abcList[0].conf.OUTPUTS_FOLDER_NAME):
            os.makedirs(_self.abcList[0].conf.OUTPUTS_FOLDER_NAME)

        # Columnas para experiments.csv
        data = {'experimentID': [], 
            'Number of Population': [], 
            'Maximum Evaluation': [], 
            'Alpha': [],
            'Beta':[],
            'Mi': [],
            'Gamma':[],
            'Vecinos': [],
            'Function': [], 
            'Dimension': [], 
            'Upper Bound': [], 
            'Lower Bound': [], 
            'isMinimize': [], 
            'Result': [], 
            'Time': []}
        experiments_df = pd.DataFrame(data)

        # Columnas para params.csv
        columns = ["experimentID"] + ["param{}".format(e) for e in range(_self.abcList[0].conf.DIMENSION)]
        params_df = pd.DataFrame(columns=columns)

        # Rellena los dataframes
        for i in range(_self.abcList[0].conf.RUN_TIME):
            # Datos del experimento
            new_row = pd.DataFrame({
                'experimentID': _self.abcList[i].experimentID,
                'Number of Population': _self.abcList[i].conf.NUMBER_OF_POPULATION,
                'Maximum Evaluation':  _self.abcList[i].conf.MAXIMUM_EVALUATION,
                'Alpha': _self.abcList[i].conf.RLIMIT,
                'Beta': _self.abcList[i].conf.BETA,
                'Mi': _self.abcList[i].conf.MI,
                'Gamma': _self.abcList[i].conf.MIN_IMPROVEMENT,
                'Vecinos': _self.abcList[i].conf.K,
                'Function': _self.abcList[i].conf.OBJECTIVE_FUNCTION.__name__,
                'Dimension': _self.abcList[i].conf.DIMENSION,
                'Upper Bound': _self.abcList[i].conf.UPPER_BOUND,
                'Lower Bound': _self.abcList[i].conf.LOWER_BOUND, 
                'isMinimize': int(_self.abcList[i].conf.MINIMIZE),
                'Result': _self.abcList[i].globalOpt,
                'Time': _self.abcList[i].globalTime}, index=[0])

            experiments_df = pd.concat([experiments_df, new_row], ignore_index=True)
            
            params_df.loc[i] = [_self.abcList[i].experimentID] + [a for a in _self.abcList[i].globalParams]
            
            # Comprueba que existe la carpeta
            if not os.path.exists(_self.abcList[i].conf.OUTPUTS_FOLDER_NAME+"/"+_self.abcList[i].conf.RESULT_BY_CYCLE_FOLDER):
                os.makedirs(_self.abcList[i].conf.OUTPUTS_FOLDER_NAME+"/"+_self.abcList[i].conf.RESULT_BY_CYCLE_FOLDER)

            # Guarda el óptimo por ciclo
            with open(_self.abcList[i].conf.OUTPUTS_FOLDER_NAME+"/"+_self.abcList[i].conf.RESULT_BY_CYCLE_FOLDER+"/"+_self.abcList[i].experimentID+".txt", 'a') as saveRes:
                for j in range(_self.abcList[i].cycle):
                    saveRes.write(str(_self.abcList[i].globalOpts[j])+"\n")
        
        params_df.to_excel("Outputs/Param_results.xlsx")
        experiments_df.to_excel("Outputs/Run_Results.xlsx")
            

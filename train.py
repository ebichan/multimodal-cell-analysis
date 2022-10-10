import optuna
import logging

from sklearn import model_selection, ensemble

def objecive(trial):
    pass

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, filename='log.txt')
    study = optuna.create_study(direction="maximize")
    study.optimize(objecive, n_trials=100)
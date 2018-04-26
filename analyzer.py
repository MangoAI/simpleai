import pandas as pd
import numpy as np
from agents.models.model import NIARKerasFeedForwardModel, NInARowData, RandomModel
from agents import NInARowTrainer
from agents.learnerAgent import compareModels
from games import ONGOING, NInARow
from tqdm import tqdm
import pickle


model = NIARKerasFeedForwardModel(3, 3, [256, 256])
model.initialize()
trainer = NInARowTrainer("data/ninarow/tictactoe/master21", model, 3, 3,
                                                      curiosity=np.sqrt(2),
                                                      max_depth=25,
                                                      trainEpochs=1000,
                                                      stochasticExploration=True, stochasticDecision=False)
allResults = []
# for i in range(1):
for i in range(len(trainer.data['modelFiles'])):
    results = []
    for j in range(len(trainer.data['modelFiles'])):
    # for j in range(1):
        m1 = trainer.loadModel(i)
        m2 = trainer.loadModel(j)
        results.append(compareModels(lambda: NInARow(3, 3), m1, m2, 100, np.sqrt(2), 25))
    allResults.append(results)
filename = 'results.pickle'
with open(filename, 'wb') as f:
    pickle.dump(allResults, f)

with open('results.pickle', 'rb') as f:
    a = pickle.load(f)
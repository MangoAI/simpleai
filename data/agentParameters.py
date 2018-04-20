class AgentParameters:

    def __init__(self, curiosity, max_depth, stochasticExploration, stochasticDecision, trainEpochs):
        self.curiosity = curiosity
        self.max_depth = max_depth
        self.stochasticExploration = stochasticExploration
        self.stochasticDecision = stochasticDecision
        self.trainEpochs = trainEpochs
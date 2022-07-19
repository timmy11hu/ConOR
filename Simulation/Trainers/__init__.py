from .single_trainer import SingleTrainer
from .pair_bootstrap import PairTrainer
from .adv_train import AdvTrainer
from .wild_bootstrap import WildTrainer
from .multiplier_bootstrap import MultiplierTrainer
from .truth_trainer import TruthTrainer
from .mc_trainer import McTrainer


def get_trainer(config, base_net, emsembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device):

    if config.method == 'Single':
        return SingleTrainer(config, base_net, emsembles, criterion, evaluation,
                       TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
    elif config.method == 'MC':
        return McTrainer(config, base_net, emsembles, criterion, evaluation,
                            TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
    elif config.method == 'Truth':
        return TruthTrainer(config, base_net, emsembles, criterion, evaluation,
                            TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
    elif config.method == 'PairBS':
        return PairTrainer(config, base_net, emsembles, criterion, evaluation,
                              TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
    elif config.method == 'AdvBS':
        return AdvTrainer(config, base_net, emsembles, criterion, evaluation,
             TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
    elif config.method == 'WildBS':
        return WildTrainer(config, base_net, emsembles, criterion, evaluation,
             TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
    elif config.method == 'MultBS':
        return MultiplierTrainer(config, base_net, emsembles, criterion, evaluation,
             TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
    else:
        raise NotImplementedError



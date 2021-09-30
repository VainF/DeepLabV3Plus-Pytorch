from .compose import IdenticalLoss, LossCompose
from .semi_loss import MI, Orth

semi_loss_zoo = {"mi": MI,
                 "orth": Orth,
                 "null": IdenticalLoss}

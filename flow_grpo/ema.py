# Copied from another repo, but I can't remember exactly which one.

from collections.abc import Iterable

import torch


class EMAModuleWrapper:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter], # theta trainable from 'default' lora adapter group
        decay: float = 0.9999, # 0.9
        update_step_interval: int = 1, # 1
        device: torch.device | None = None, # device(type='cuda', index=0)
    ):
        import ipdb; ipdb.set_trace()
        parameters = list(parameters)
        self.ema_parameters = [p.clone().detach().to(device) for p in parameters] # NOTE a copy of theta

        self.temp_stored_parameters = None

        self.decay = decay
        self.update_step_interval = update_step_interval
        self.device = device

    def get_current_decay(self, optimization_step) -> float:
        import ipdb; ipdb.set_trace()
        return min((1 + optimization_step) / (10 + optimization_step), self.decay)
        # e.g., optimization_step=1, return is min(2/11, 0.9)=2/11=0.1818...

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step):
        import ipdb; ipdb.set_trace()
        parameters = list(parameters) # 本来就是list, 这里没有变化

        one_minus_decay = 1 - self.get_current_decay(optimization_step) # 1 - 0.1818... = 0.8181...
        # use theta=parameters to update self.ema_parameters=theta^old!
        if (optimization_step + 1) % self.update_step_interval == 0: # 2 % 1 == 0
            for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
                if parameter.requires_grad: # parameter.requires_grad=True, ema_parameter.requires_grad=False
                    if ema_parameter.device == parameter.device:
                        ema_parameter.add_(one_minus_decay * (parameter - ema_parameter)) 
                        # NOTE theta^old <= theta^old + (1-eta_i) * (theta - theta^old) 
                        #                <= theta^old + (1-eta_i) * theta + (eta_i - 1) * theta^old
                        #                <= eta_i * theta^old + (1-eta_i) * theta
                        #                <= 0.18  * theta^old + 0.82      * theta; NOTE 这里0.18代表对于已有的theta^old的保留的量；0.82权重则是加到新的参数theta的权重
                    else:
                        # in place calculations to save memory
                        parameter_copy = parameter.detach().to(ema_parameter.device)
                        parameter_copy.sub_(ema_parameter)
                        parameter_copy.mul_(one_minus_decay)
                        ema_parameter.add_(parameter_copy)
                        del parameter_copy

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        import ipdb; ipdb.set_trace()
        self.device = device
        self.ema_parameters = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.ema_parameters
        ]

    @torch.no_grad()
    def sync_with_model(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Force the EMA parameters to be a direct copy of the given model parameters.
        This is used to create a snapshot for the rollout policy.
        """
        import ipdb; ipdb.set_trace()
        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            ema_parameter.data.copy_(parameter.detach().data)

    def copy_ema_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True, grad=False) -> None:
        import ipdb; ipdb.set_trace()
        if store_temp: # True
            if grad: # False
                self.temp_stored_parameters = [parameter.data.clone() for parameter in parameters]
            else:
                self.temp_stored_parameters = [parameter.detach().cpu() for parameter in parameters] # NOTE here, theta 'default' lora adapter group 这是把parameter输入参数集合，保存到self ema的临时cache = self.temp_stored_parameters

        parameters = list(parameters) # theta 'default' lora adapter, trainable
        for ema_parameter, parameter in zip(self.ema_parameters, parameters, strict=True):
            parameter.data.copy_(ema_parameter.to(parameter.device).data) # NOTE 这是把ema_parameter的数据，复制到parameter的存储空间, parameter.data的memory不变，只是数值被覆盖

    def copy_temp_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        import ipdb; ipdb.set_trace()
        for temp_parameter, parameter in zip(self.temp_stored_parameters, parameters, strict=True):
            # Ensure the temp parameter is on the right device
            parameter.data.copy_(temp_parameter.to(parameter.device)) # ema.temp_stored_parameters --> theta

        self.temp_stored_parameters = None

    def load_state_dict(self, state_dict: dict) -> None:
        import ipdb; ipdb.set_trace()
        self.decay = self.decay if self.decay else state_dict.get("decay", self.decay)
        self.ema_parameters = state_dict.get("ema_parameters")
        self.to(self.device)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ema_parameters": self.ema_parameters,
        }

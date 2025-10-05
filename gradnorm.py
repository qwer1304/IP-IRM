import torch
import torch.nn.functional as F

class GradNormLossBalancer:
    def __init__(self, initial_weights, alpha=1.2, device='cpu', smoothing=False, tau=None, eps=1e-8):
        """
        Args:
            initial_weights (dict): Initial task weights, e.g., {'cont': 1.0, 'keep_cont': 1.0, 'penalty': 1.0}
            alpha (float): Moving average smoothing factor for task loss rates.
            smoothing (bool): False - original rates, True - moving average w/ alpha
            tau (float): loss rates divisor, lower value -> higher effective loss rate -> lower true loss rate -> higher learning rate
        """
        self.task_weights = {
            k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, requires_grad=True, device=device,))
            for k, v in initial_weights.items()
        }
        self.task_names = list(initial_weights.keys())
        self.alpha = alpha
        self.initial_losses = {}
        self.running_loss_rates = {k: 1.0 for k in self.task_names}  # Initialized to 1.0
        self.smoothing = smoothing
        self.device = device
        if tau is None:
            tau = [1.0 for k in self.task_names]
        else:
            mtau = sum([v for v in tau.values()]) / len(self.task_names)
            tau = [v / mtau for v in tau.values()] 
        self.tau = torch.tensor(tau, device=device, requires_grad=False)
        self.eps = eps

    def reset_weights(self, new_initial_weights):
        for k, new_val in new_initial_weights.items():
            if k not in self.task_weights:
                raise ValueError(f"Task '{k}' not found in existing task_weights.")
            with torch.no_grad():
                self.task_weights[k].copy_(torch.tensor(new_val, device=self.device))

        # Optionally reset other internal state
        self.initial_losses = {}
        self.running_loss_rates = {k: 1.0 for k in self.task_names}

    # to be called AFTER weights update by optimizer, BEFORE next iteration
    def clamp_weights(self, lb=None, ub=None): 
        """
        lb, ub : dictionaries of lb and ub constraints on task's weight
        """
        for k in self.task_weights.keys():
            x   = self.task_weights[k]
            llb = lb[k] if k in lb else None
            uub = ub[k] if k in ub else None
            x = torch.clamp(x, llb, uub)
            if x != self.task_weights[k]:    
                with torch.no_grad():
                   self.task_weights[k].copy_(x.to(self.task_weights[k].device))

    def parameters(self):
        # So you can pass these to the optimizer
        return list(self.task_weights.values())

    def compute_weights_and_loss(self, losses_dict, grad_norms):
        """
        Args:
            losses_dict (dict): Mapping from task name to loss tensor (scalar), not weighted by task's weight.
            grad_norms (dict):  Mapping from task name to sum of its parameters gradient norms (scalar tensor), 
                                not weighted by task's weight
        
        Returns:
            dict: weight for each task
            torch.Tensor: gradnorm_loss
        Notes:
            1. It's expected that weights are updated through an optimizer on gradnorm_loss:
                optimizer.zero_grads()
                gradnorm_loss.backward()
                optimizer.step()
               w/ the optimizer has been given the weights to optimize
            2. Returned weights are normalized to sum to 1 and detached and should be used to combine task losses
               into aggregate loss before optimizer is applied on model's parameters
        """

        """
        extract losses into list; order determined by task_names
        task_names is a list of names extracted from keys of inital weights dict
        """
        task_losses = [losses_dict[k] for k in self.task_names]
                                                               
        # Step 1: Store initial losses if not done
        for i, name in enumerate(self.task_names):
            if name not in self.initial_losses:
                self.initial_losses[name] = task_losses[i].detach()

        # Step 2: Compute gradient norms of each task loss w/ unnormalized weights
        weights = torch.stack([self.task_weights[k] for k in self.task_names])
        grad_norms = torch.stack([grad_norms[k] for k in self.task_names])
        weighted_grad_norms = grad_norms * weights
        avg_grad_norm = weighted_grad_norms.mean()

        # Step 3: Compute inverse training rates
        loss_ratios = torch.stack([losses_dict[k] / self.initial_losses[k] for k in self.task_names])

        normalized_ratios = loss_ratios / (loss_ratios.mean().detach() + self.eps)
        # smaller tau -> bigger effective loss_rates; since the objective is to have similar loss rates, 
        # this'd cause the true loss rate to decrease 
        loss_rates = normalized_ratios / self.tau 
        
        if not self.smoothing:        
            # Step 4a: Update running rates (instantenous, original)
            loss_rates = loss_rates ** self.alpha
            smoothed_rates = loss_rates
        else:
            # Step 4b: Update running rates (smoothed)
            for i, k in enumerate(self.task_names):
                self.running_loss_rates[k] = (
                    self.alpha * self.running_loss_rates[k] + (1 - self.alpha) * loss_rates[i].item()
                )
            smoothed_rates = torch.tensor([self.running_loss_rates[k] for k in self.task_names], device=grads.device)

        # Step 5: GradNorm loss
        """
        UNNORMALIZED weights!
        If they normalized first, the constraint sum_i w_i = const would kill the degrees of freedom that the optimizer needs 
        to learn relative scales. The GradNorm loss must be unconstrained, otherwise the model can't freely adjust magnitudes.
        Normalization is only applied after the update, when you want to use the weights to combine task losses in the forward pass.
        """
        gradnorm_loss = (weighted_grad_norms - avg_grad_norm * smoothed_rates).abs().sum()
        #gradnorm_loss = ((weighted_grad_norms - avg_grad_norm * smoothed_rates) ** 2).sum()

        # Step 6: Normalize task weights
        # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain 
        # the output of a machine to always be positive.
        raw_weights = F.softplus(torch.stack([v for v in self.task_weights.values()]))
        weights_sum = raw_weights.sum()
        # normalized to sum to the number of tasks
        normed_weights = len(self.task_names) * raw_weights / weights_sum
        # DETACHED!
        normalized_weights = {k: normed_weights[i].detach().to(self.device) \
                for i, k in enumerate(self.task_names)
        }
        #print(raw_weights, normalized_weights)
        return normalized_weights, gradnorm_loss, grad_norms

    def state_dict(self):
        return {
            "task_weights": {k: v.detach() for k, v in self.task_weights.items()},
            "initial_losses": {k: v for k, v in self.initial_losses.items()},
            "running_loss_rates": self.running_loss_rates,
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict["task_weights"].items():
            self.task_weights[k] = torch.nn.Parameter(v.clone().requires_grad_())
        self.initial_losses = {
            k: v.clone() for k, v in state_dict["initial_losses"].items()
        }
        self.running_loss_rates = state_dict["running_loss_rates"]

    def set_alpha(self, alpha):
        self.alpha = alpha
        

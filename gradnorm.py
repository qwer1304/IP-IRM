import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
import collections

class GradNormLossBalancer(nn.Module):
    def __init__(self, initial_weights, alpha=1.2, device='cpu', smoothing=False, tau=None, eps=1e-8, debug=None, 
                    beta=1.0, Gscaler=1.0, avgG_detach_frac=0.0, gradnorm_loss_type='L1', gradnorm_lr=1e-3, gradnorm_loss_lambda=5e-4):
        """
        Args:
            initial_weights (dict): Initial task weights, e.g., {'cont': 1.0, 'keep_cont': 1.0, 'penalty': 1.0}
            alpha (float): Moving average smoothing factor for task loss rates.
            smoothing (bool): False - original rates, True - moving average w/ alpha
            tau (dict): loss rates divisors, lower value -> weight increases
            Note: initial_weights keys determine the tasks to be tracked by GradNorm
        """
        super().__init__()

        # looks and behaves like a dict, but parameters are registered

        self.task_weights = nn.ParameterDict({
            k: nn.Parameter(
                torch.as_tensor(v, dtype=torch.float32, device=device).clone().detach().requires_grad_()
            )
            for k, v in initial_weights.items()
        })

        self.task_names = list(initial_weights.keys())
        self.alpha = alpha
        self.initial_losses = {}
        self.running_loss_rates = {k: 1.0 for k in self.task_names}  # Initialized to 1.0
        self.smoothing = smoothing
        self.device = device
        self.tau = self.set_tau(tau)
        self.eps = eps
        self.debug = debug
        self.beta = beta
        self.Gscaler = Gscaler
        self.avgG_detach_frac = avgG_detach_frac
        self.gradnorm_loss_type = gradnorm_loss_type
        self.gradnorm_lr = gradnorm_lr
        self.gradnorm_loss_lambda = gradnorm_loss_lambda
        
        # --- persistent state for pathological state detection
        # --- configurable thresholds ---
        self.window_size     = 5        # number of consecutive batches to monitor
        self.required_count  = 4        # how many in the window must be all-negative
        self.min_mag         = 1e-4     # ignore tiny grad values as numerical noise
        self.cooldown_period = 2 * self.window_size

        self.gn_bad_buffer            = collections.deque(maxlen=self.window_size)
        self.gn_last_mitigation_batch = -9999
        self.batch_idx                = 0

        self.w                        = 0

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

    def compute_weights_and_loss(self, losses_dict, grad_norms):
        """
        Objective:  GradNorm should establish a common scale for gradient magnitudes, and also should balance
                    training rates of different tasks. 
        Method:     The common scale for gradients is the average gradient norm, G(t), which establishes 
                    a baseline at each timestep t by which we can determine relative gradient sizes. 
                    The relative inverse training rate of task i, r_i(t), can be used to rate-balance the gradients. 
                    The higher the value of r_i(t) (slower training rate), the higher the gradient magnitudes 
                    should be for task i in order to encourage the task to train more quickly.
        Training objective: GN_loss(t; w_i(t)) = sum_i L1[G_i(t; w_i(t)) - avg_i G(t; w_i(t)) * r_i(t)^alpha]
                            G_i(t; w_i(t)) = L2[d/dTheta (w_i(t)*L_i(t))]
                            r_i(t) = normedL_i(t) / avg_i normedL_i(t)
                            normedL_i(t) = L_i(t) / L_i(0)
        Trained parameters: w_i(t)
        Hyperparametrs: alpha
        Args:
            losses_dict (dict): Mapping from task name to loss tensor (scalar), not weighted by task's weight.
            grad_norms (dict):  Mapping from task name to sum of its parameters gradient norms (scalar tensor), 
                                not weighted by task's weight
        
        Returns:
            dict: weight for each task
            torch.Tensor: gradnorm_loss
            torch.tensor: rate for each task (raised to alpha power)
        Notes:
            1. It's expected that weights are updated through an optimizer on gradnorm_loss:
                optimizer.zero_grads()
                gradnorm_loss.backward()
                optimizer.step()
               w/ the optimizer has been given the weights to optimize
            2. Returned weights are normalized to sum to num_tasks and detached and should be used 
               to combine task losses into aggregate loss before optimizer is applied on model's parameters
            3. First time this method is run the losses are used to set the initial losses
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

        # --- SOFT-DETACH avgG: keep some gradient flow, but damp the dominating global term ---
        # avg_grad_norm is a scalar with grad. We produce a mixed scalar that partially detaches.
        avgG_semi_detached = (1.0 - self.avgG_detach_frac) * avg_grad_norm + \
                             (0.0 + self.avgG_detach_frac) * avg_grad_norm.detach()

        # Step 3: Compute inverse training rates
        loss_ratios = torch.stack([losses_dict[k] / self.initial_losses[k] for k in self.task_names])

        normalized_ratios = loss_ratios / (loss_ratios.mean().detach() + self.eps)
        # smaller tau -> bigger target loss_rates; since the objective is to have similar loss rates, 
        # this'd cause the weight to increase 
        loss_rates = normalized_ratios / self.tau 
        
        # Clamp excessive rates to prevent runaway domination
        R_max = getattr(self, "max_rate", 3.0)
        loss_rates = torch.clamp(loss_rates, max=R_max)        
        
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
        gradnorm_loss  = self.Gscaler * (weighted_grad_norms - avgG_semi_detached * smoothed_rates)
        gradnorm_loss  = gradnorm_loss.abs() if self.gradnorm_loss_type == 'L1' else (gradnorm_loss ** 2)
        gradnorm_loss  = gradnorm_loss.mean()
        # leave this for now, but don't instantiate - gradnorm_loss_lambda = 0.
        V = weights.sum()
        gradnorm_loss += self.gradnorm_loss_lambda * (V.log() - math.log(len(self.task_names)))**2
        
        # Step 6: Normalize task weights
        # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain 
        # the output of a machine to always be positive.
        task_weights = torch.stack([v for v in self.task_weights.values()])
        raw_weights = F.softplus(task_weights, beta=self.beta)
        weights_sum = raw_weights.sum()
        # normalized to sum to the number of tasks
        normed_weights = len(self.task_names) * raw_weights / weights_sum
        # DETACHED!
        normalized_weights = {k: normed_weights[i].detach().to(self.device) \
                for i, k in enumerate(self.task_names)
        }
        
        # Diagnostics / safeguard code against pathological misconfiguration / behavior
        small_eps = 1e-6
        T = len(self.task_names)
        veights = weights.detach()       # unnormalized v
        g = grad_norms.detach()          # grad norms g_i
        avgG = avgG_semi_detached.detach()
        rates = smoothed_rates.detach()  

        r = (veights * g) - (avgG * rates)           # residuals r_i
        if self.gradnorm_loss_type == 'L1':
            s = r.sign()                             # s_i = sign(res_i)
            global_term = (s * rates).mean()         # (1/T) sum_i s_i * rho_i
            expected_v_grad = self.Gscaler * 1.0 * g * (s - (1.0 - self.avgG_detach_frac) * global_term) / T
        elif self.gradnorm_loss_type == 'L2':
            global_term = (r * rates).mean()         # (1/T) sum_j r_j * rate_j
            expected_v_grad = self.Gscaler * 2.0 * g * (r - (1.0 - self.avgG_detach_frac) * global_term) / T
        expected_v_grad = expected_v_grad.detach().cpu()

        #veights_ratios = veights / veights.sum()
        #dr = veights_ratios.diff()

        # --- GradNorm pathological state detector ---

        # --- evaluate current batch condition ---
        # ignore elements with very small magnitude
        significant_mask = expected_v_grad.abs() > self.min_mag

        # diagnostic booleans
        all_negative = (expected_v_grad[significant_mask] < 0).all()
        all_positive = (expected_v_grad[significant_mask] > 0).all()
        mixed        = not (all_negative or all_positive)

        this_batch_bad = 0
        if significant_mask.sum() > 0:
            # pathological if *all* significant 'expected_v_grad' are negative.
            # sometimes also when they're all positive
            if all_negative:
                this_batch_bad = -1
            if all_positive:
                E = expected_v_grad[significant_mask].cpu().detach().numpy()
                w = (veights / veights.sum()).cpu().numpy()
                delta_w_norm = np.linalg.norm(w - self.w)
                eq_metric = np.std(E) / (abs(E.mean()) + 1e-8)
                w_prev = self.w 
                self.w = w
                if (eq_metric > 0.1) or (delta_w_norm > 1e-3):
                    this_batch_bad = 1

        # store in rolling window
        self.gn_bad_buffer.append(this_batch_bad)

        # determine if persistent pathology
        if len(self.gn_bad_buffer) == self.window_size:
            count_bad = sum(self.gn_bad_buffer)
            persistent_bad = (abs(count_bad) >= self.required_count)
        else:
            persistent_bad = False

        # --- mitigation (only once per cooldown) ---
        if persistent_bad:
            msg_bad = 'all-negative' if count_bad < 0 else 'all-positive'
            warnings.warn(f"[GN WARNING] Persistent {msg_bad} detected: ({count_bad}/{self.window_size})")

            """
            DON'T APLLY MITIGATION YET!!!!!!!!!!!
            if (self.batch_idx - self.gn_last_mitigation_batch) > self.cooldown_period:
                # Option A: reduce GN learning rate
                for gparam_group in gn_optimizer.param_groups:
                    gparam_group['lr'] *= 0.5
                print(f"  -> gn_optimizer.lr *= 0.5 "
                      f"(now {gn_optimizer.param_groups[0]['lr']})")

                # Option B (alternative): halve Gscaler
                # Gscaler *= 0.5

                # Option C (alternative): double tau_p
                # tau_p *= 2
                self.gn_last_mitigation_batch = batch_idx
            """

        self.batch_idx += 1

        if self.debug and 'gn' in self.debug:
            with np.printoptions(precision=6):
                # convert to numpy to use numpy's formatting options
                print()
                print("tasks:\t\t", self.task_names)
                print("weights (pars):\t", veights.cpu().numpy())
                print("g (grad norms):\t", g.cpu().numpy())
                print("avgG:\t\t", avgG.cpu().numpy())
                print("rates:\t\t", rates.cpu().numpy())
                print("residuals r:\t", r.cpu().detach().numpy())
                print("global_term:\t", global_term.cpu().numpy())
                print("expected_v_grad:", expected_v_grad.numpy())
                print("normed_weights:\t", np.array([normalized_weights[k].cpu().item() for k in self.task_names]))
                print("gradnorm_loss:\t", gradnorm_loss.cpu().detach().numpy())
                print(f"all_neg {all_negative.numpy()} all_pos {all_positive.numpy()} mixed {mixed}" +
                      f" sgnfcnt_msk {np.array(significant_mask.tolist())} prsst_bad {persistent_bad}")
                if all_positive:
                    print(f"eqlbrm/rnwy dtct: w {w} w_prev {w_prev} |w-w_p| {delta_w_norm} eq_metric {eq_metric}")

        return normalized_weights, gradnorm_loss, smoothed_rates

    # --------------------------------------------
    # Custom state_dict for GradNorm-specific data
    # --------------------------------------------
    def state_dict(self):
        # Use .detach() to avoid saving gradients
        return {
            "task_weights": {k: v.detach().clone() for k, v in self.task_weights.items()},
            "initial_losses": {k: v.clone().detach() for k, v in self.initial_losses.items()},
            "running_loss_rates": {k: float(v) for k, v in self.running_loss_rates.items()},
        }

    def load_state_dict(self, state_dict, optimizer=None, gradnorm_optimizer=None, add_param_groups_if_new=False):
        """
        Load saved state. If optimizer(s) are provided, this function will:
          - copy into existing parameters in-place when possible (preserves optimizer tracking)
          - if new params must be created, it will register them, and optionally add them
            to the provided gradnorm_optimizer (via add_param_group).
        Returns: list of newly_registered_param_names
        """
        newly_registered = []

        # task_weights: update in-place if present, otherwise register new Parameter
        for k, v in state_dict["task_weights"].items():
            v = v.detach().clone()
            if k in self.task_weights:
                # in-place copy preserves object identity (optimizer still tracks it)
                self.task_weights[k].data.copy_(v.to(self.task_weights[k].device))
            else:
                # register new param
                self.task_weights[k] = nn.Parameter(v.clone().detach().requires_grad_())
                newly_registered.append(k)

        # initial_losses and running rates (non-Parameters)
        self.initial_losses = {
            k: (v.clone().detach() if torch.is_tensor(v) else torch.tensor(v))
            for k, v in state_dict["initial_losses"].items()
        }
        self.running_loss_rates = dict(state_dict.get("running_loss_rates", {}))

        # If new params were registered and a gradnorm_optimizer is provided, add them
        if newly_registered and gradnorm_optimizer is not None and add_param_groups_if_new:
            new_params = [self.task_weights[k] for k in newly_registered]
            gradnorm_optimizer.add_param_group({'params': new_params})

        return newly_registered
        
    def set_tau(self, tau):
        if tau is None:
            tau = [1.0 for k in self.task_names]
        else:
            tau = [tau[k] for k in self.task_names]            
        self.tau = torch.tensor(tau, device=self.device, dtype=torch.float, requires_grad=False)

    def rescale_weights(self):
        for k, v in self.task_weights.items():
            v = v.detach().clone()
            vsum = v.sum()
            v = v * len(self.task_weights) / (vsum + 1e-12)
            print()
            print(v, vsum, len(self.task_weights))
            if k in self.task_weights:
                # in-place copy preserves object identity (optimizer still tracks it)
                self.task_weights[k].data.copy_(v.to(self.task_weights[k].device))

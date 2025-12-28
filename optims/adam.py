import numpy as np


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_iteration=30, scheduler='flat'):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None  # first moment
        self.v = None  # second moment

        self.n_iteration = n_iteration

        # learning rate scheduler callable: f(step:int)->float
        self.scheduler = self.__get_scheduler(scheduler)
        self.t = 0     # timestep

    def __get_scheduler(self, scheduler_spec):
        """
        Build a scheduler function f(t) -> lr_t based on specification.

        scheduler_spec can be either a string among {"flat", "exponent", "sinusoidal", "onecycle"}
        or a dict with a required key 'type' and optional parameters:
          - exponent: {'type': 'exponent', 'gamma': float}
          - sinusoidal: {'type': 'sinusoidal', 'min_lr': float, 'max_lr': float, 'period': int}
          - onecycle (cosine): {
                'type': 'onecycle',
                'max_lr': float,
                'total_steps': int,
                'pct_start': float,            # fraction of steps to increase
                'div_factor': float,           # initial_lr = max_lr / div_factor
                'final_div_factor': float,     # final_lr = initial_lr / final_div_factor
                'anneal_strategy': 'cosine'    # currently supports only 'cosine'
            }
        Defaults are chosen if optional parameters are not provided.
        """
        # Normalize input
        if isinstance(scheduler_spec, str):
            sched_type = scheduler_spec.lower()
            params = {}
        elif isinstance(scheduler_spec, dict):
            sched_type = str(scheduler_spec.get('type', 'flat')).lower()
            params = dict(scheduler_spec)
        else:
            sched_type = 'flat'
            params = {}

        if sched_type == 'flat':
            def flat_schedule(t):
                return self.lr

            return flat_schedule

        if sched_type in ('exponent', 'exponential', 'exp'):
            # gamma < 1 yields decay per step. Default to mild decay.
            gamma = float(params.get('gamma', 0.99))

            def exp_schedule(t):
                # use (t-1) to start from base lr at first step
                return self.lr * (gamma ** max(t - 1, 0))

            return exp_schedule

        if sched_type in ('sin', 'sine', 'cosine', 'sinusoidal'):
            # Oscillate between min_lr and max_lr over 'period' steps (default: n_iteration)
            max_lr = float(params.get('max_lr', self.lr))
            min_lr = float(params.get('min_lr', 0.1 * self.lr))
            period = int(params.get('period', max(self.n_iteration, 1)))
            amp = 0.5 * (max_lr - min_lr)
            mid = min_lr + amp

            def sin_schedule(t):
                # cosine schedule: starts at max_lr at t=1
                # map t to [0, 2*pi] over one period
                x = 2.0 * np.pi * (max(t - 1, 0) % period) / max(period, 1)
                return mid + amp * np.cos(x)

            return sin_schedule

        if sched_type in ('onecycle', 'one_cycle', 'onecyclelr', 'one_cycle_lr'):
            # OneCycleLR with cosine annealing (similar to PyTorch OneCycleLR)
            # Parameters and defaults inspired by PyTorch:
            max_lr = float(params.get('max_lr', self.lr))
            total_steps = int(params.get(
                'total_steps', max(self.n_iteration, 1)))
            pct_start = float(params.get('pct_start', 0.3))
            div_factor = float(params.get('div_factor', 25.0))
            final_div_factor = float(params.get('final_div_factor', 1e4))
            anneal_strategy = str(params.get(
                'anneal_strategy', 'cosine')).lower()

            # We currently support only cosine as requested
            if anneal_strategy not in ('cos', 'cosine'):
                anneal_strategy = 'cosine'

            # Derive initial and final learning rates
            initial_lr = max_lr / max(div_factor, 1e-12)
            final_lr = initial_lr / max(final_div_factor, 1e-12)

            # Step partitioning
            total_steps = max(total_steps, 1)
            step_size_up = int(round(pct_start * total_steps))
            step_size_up = min(max(step_size_up, 1),
                               total_steps)  # ensure at least 1
            step_size_down = max(total_steps - step_size_up, 0)

            def _anneal_cos(start, end, pct):
                # cosine annealing between start and end with pct in [0,1]
                return end + 0.5 * (start - end) * (1 + np.cos(np.pi * np.clip(pct, 0.0, 1.0)))

            def onecycle_schedule(t):
                # t is 1-based in this optimizer; convert to 0-based index
                s = max(t - 1, 0)
                if s >= total_steps:
                    # after the cycle, keep final lr
                    return final_lr
                if step_size_down == 0 or s < step_size_up:
                    # warm-up (increase from initial to max)
                    pct = s / max(step_size_up, 1)
                    return _anneal_cos(initial_lr, max_lr, pct)
                # cooldown (decrease from max to final)
                pct = (s - step_size_up) / max(step_size_down, 1)
                return _anneal_cos(max_lr, final_lr, pct)

            return onecycle_schedule

        # Fallback to flat if unknown
        def default_schedule(t):
            return self.lr

        return default_schedule

    def step(self, params: np.ndarray, grads: np.ndarray):
        """
        params: list of floats (parameters to update)
        grads:  list of floats (gradients dL/dparam with same length as params)
        returns: updated params (list of floats)
        """
        if self.m is None or self.v is None:
            # lazy initialization
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        # apply scheduled learning rate for this step
        lr_t = self.scheduler(
            self.t) if self.scheduler is not None else self.lr
        print("Current Learning Rate: ", lr_t)
        params = params - lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
        return params

import math


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_iteration=30):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None  # first moment
        self.v = None  # second moment

        self.n_iteration = n_iteration

        self.t = 0     # timestep

    def step(self, params, grads):
        """
        params: list of floats (parameters to update)
        grads:  list of floats (gradients dL/dparam with same length as params)
        returns: updated params (list of floats)
        """
        if self.m is None or self.v is None:
            # lazy initialization
            self.m = [0.0 for _ in params]
            self.v = [0.0 for _ in params]

        self.t += 1
        lr_t = self.lr

        new_params = []
        for i, (theta, g) in enumerate(zip(params, grads)):
            # update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            # update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            # bias-corrected moments
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # parameter update
            theta = theta - lr_t * \
                m_hat / (math.sqrt(v_hat) + self.eps)
            new_params.append(theta)

        return new_params

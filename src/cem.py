import torch


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class CEMOptimizer(Optimizer):
    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (torch.Tensor): An array of upper bounds
            lower_bound (torch.Tensor): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim = sol_dim
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.ub = upper_bound
        self.lb = lower_bound
        self.epsilon = epsilon
        self.alpha = alpha
        self.cost_function = cost_function

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (torch.Tensor): The mean of the initial candidate distribution.
            init_var (torch.Tensor): The variance of the initial candidate distribution.
        """
        mean = init_mean.clone()
        var = init_var.clone()
        t = 0

        while (t < self.max_iters) and torch.max(var) > self.epsilon:
            lb_dist = mean - self.lb
            ub_dist = self.ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)

            samples = torch.normal(mean.expand(self.popsize, -1), torch.sqrt(constrained_var).expand(self.popsize, -1))

            costs = self.cost_function(samples)
            sorted_costs, indices = torch.sort(costs)
            elites = samples[indices[:self.num_elites]]

            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, dim=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        return mean

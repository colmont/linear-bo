"""
Rover task.

Adapted from:

- https://github.com/zi-w/Ensemble-Bayesian-Optimization/blob/master/test_functions/rover_utils.py
- https://github.com/DonneyF/bo-playground/blob/main/bo/objectives/rover.py
"""

import concurrent.futures

import numpy as np
import scipy.interpolate as si
import torch
from botorch.test_functions.base import BaseTestProblem
from jaxtyping import Float


class Trajectory:
    """TODO"""

    def __init__(self):
        pass

    def set_params(self, start, goal, params):
        """TODO"""
        raise NotImplementedError

    def get_points(self, t):
        """TODO"""
        raise NotImplementedError

    @property
    def param_size(self):
        """TODO"""
        raise NotImplementedError


class PointBSpline(Trajectory):
    """
    :param dim: number of dimensions of the state space
    :param num_points: number of internal points used to represent the trajectory.
    """

    def __init__(self, dim, num_points):
        self.tck = None
        self.d = dim
        self.npoints = num_points

    def set_params(self, params, start=None, goal=None):
        """
        Set fit the parameters of the spline from a set of points. If values are given for start or goal,
        the start or endpoint of the trajectory will be forces on those points, respectively.

        (TODO: add more docstring)
        """
        points = params.reshape((-1, self.d)).T

        if start is not None:
            points = np.hstack((start[:, None], points))

        if goal is not None:
            points = np.hstack((points, goal[:, None]))

        self.tck, u = si.splprep(points, k=3)

        if start is not None:
            for a, sv in zip(self.tck[1], start):
                a[0] = sv

        if goal is not None:
            for a, gv in zip(self.tck[1], goal):
                a[-1] = gv

    def get_points(self, t):
        """TODO"""
        assert self.tck is not None, "Parameters have to be set with set_params() before points can be queried."
        return np.vstack(si.splev(t, self.tck)).T

    @property
    def param_size(self):
        """TODO"""
        return self.d * self.npoints


def simple_rbf(x, point):
    """TODO"""
    return 1 - np.exp(-np.sum(((x - point) / 0.25) ** 2))


class RoverDomain:
    """
    Rover domain defined on :math:`R^d`

    :param cost_fn: vectorized function giving a scalar cost to states
    :param start: a start state for the rover
    :param goal: a goal state
    :param traj: a parameterized trajectory object offering an interface
        to interpolate point on the trajectory
    :param s_range: the min and max of the state with s_range[0] in R^d are
        the mins and s_range[1] in R^d are the maxs

    (TODO: add more docstring)
    """

    def __init__(
        self,
        cost_fn,
        start: Float[np.ndarray, " d"],
        goal: Float[np.ndarray, " d"],
        traj,
        s_range,
        start_miss_cost=None,
        goal_miss_cost=None,
        force_start=True,
        force_goal=True,
        only_add_start_goal=True,
        rnd_stream=None,
    ):
        self.cost_fn = cost_fn
        self.start = start
        self.goal = goal
        self.traj = traj
        self.s_range = s_range
        self.rnd_stream = rnd_stream
        self.force_start = force_start
        self.force_goal = force_goal

        self.goal_miss_cost = goal_miss_cost
        self.start_miss_cost = start_miss_cost

        if self.start_miss_cost is None:
            self.start_miss_cost = simple_rbf
        if self.goal_miss_cost is None:
            self.goal_miss_cost = simple_rbf

        if self.rnd_stream is None:
            self.rnd_stream = np.random.RandomState(np.random.randint(0, 2**32 - 1))

    # return the negative cost which need to be optimized
    def __call__(self, params, n_samples=1000):
        """TODO"""
        self.set_params(params)

        return -self.estimate_cost(n_samples=n_samples)

    def set_params(self, params):
        """TODO"""
        self.traj.set_params(
            params + self.rnd_stream.normal(0, 1e-4, params.shape),
            self.start if self.force_start else None,
            self.goal if self.force_goal else None,
        )

    def estimate_cost(self, n_samples=1000):
        """TODO"""
        # get points on the trajectory
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))
        # compute cost at each point
        costs = self.cost_fn(points)

        # estimate (trapezoidal) the integral of the cost along traj
        avg_cost = 0.5 * (costs[:-1] + costs[1:])
        ell = np.linalg.norm(points[1:] - points[:-1], axis=1)
        total_cost = np.sum(ell * avg_cost)

        if not self.force_start:
            total_cost += self.start_miss_cost(points[0], self.start)
        if not self.force_goal:
            total_cost += self.goal_miss_cost(points[-1], self.goal)

        return total_cost

    @property
    def input_size(self):
        """TODO"""
        return self.traj.param_size


class AABoxes:
    """TODO"""

    def __init__(self, lows, highs):
        self.l = lows
        self.h = highs

    def contains(self, X):
        """TODO"""
        if X.ndim == 1:
            X = X[None, :]

        lX = self.l.T[None, :, :] <= X[:, :, None]
        hX = self.h.T[None, :, :] > X[:, :, None]

        return lX.all(axis=1) & hX.all(axis=1)


class NegGeom:
    """TODO"""

    def __init__(self, geometry):
        self.geom = geometry

    def contains(self, X):
        """TODO"""
        return ~self.geom.contains(X)


class UnionGeom:
    """TODO"""

    def __init__(self, geometries):
        self.geoms = geometries

    def contains(self, X):
        """TODO"""
        return np.any(np.hstack([g.contains(X) for g in self.geoms]), axis=1, keepdims=True)


class ConstObstacleCost:
    """TODO"""

    def __init__(self, geometry, cost):
        self.geom = geometry
        self.c = cost

    def __call__(self, X):
        """TODO"""
        return self.c * self.geom.contains(X)


class ConstCost:
    """TODO"""

    def __init__(self, cost):
        self.c = cost

    def __call__(self, X):
        """TODO"""
        if X.ndim == 1:
            X = X[None, :]
        return np.ones((X.shape[0], 1)) * self.c


class AdditiveCosts:
    """TODO"""

    def __init__(self, fns):
        self.fns = fns

    def __call__(self, X):
        """TODO"""
        return np.sum(np.hstack([f(X) for f in self.fns]), axis=1)


class GMCost:
    """TODO"""

    def __init__(self, centers, sigmas, weights=None):
        self.c = centers
        self.s = sigmas
        if self.s.ndim == 1:
            self.s = self.s[:, None]
        self.w = weights
        if weights is None:
            self.w = np.ones(centers.shape[0])

    def __call__(self, X):
        """TODO"""
        if X.ndim == 1:
            X = X[None, :]

        return np.exp(-np.sum(((X[:, :, None] - self.c.T[None, :, :]) / self.s.T[None, :, :]) ** 2, axis=1)).dot(self.w)


class NormalizedInputFn:
    """TODO"""

    def __init__(self, fn_instance, x_range):
        self.fn_instance = fn_instance
        self.x_range = x_range

    def __call__(self, x):
        """TODO"""
        return self.fn_instance(self.project_input(x))

    def project_input(self, x):
        """TODO"""
        return x * (self.x_range[1] - self.x_range[0]) + self.x_range[0]

    def inv_project_input(self, x):
        """TODO"""
        return (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

    def get_range(self):
        """TODO"""
        return np.array([np.zeros(self.x_range[0].shape[0]), np.ones(self.x_range[0].shape[0])])


class ConstantOffsetFn:
    """TODO"""

    def __init__(self, fn_instance, offset):
        self.fn_instance = fn_instance
        self.offset = offset

    def __call__(self, x):
        """TODO"""
        return self.fn_instance(x) + self.offset

    def get_range(self):
        """TODO"""
        return self.fn_instance.get_range()


def create_cost_large():
    """TODO"""
    c = np.array(
        [
            [0.43143755, 0.20876147],
            [0.38485367, 0.39183579],
            [0.02985961, 0.22328303],
            [0.7803707, 0.3447003],
            [0.93685657, 0.56297285],
            [0.04194252, 0.23598362],
            [0.28049582, 0.40984475],
            [0.6756053, 0.70939481],
            [0.01926493, 0.86972335],
            [0.5993437, 0.63347932],
            [0.57807619, 0.40180792],
            [0.56824287, 0.75486851],
            [0.35403502, 0.38591056],
            [0.72492026, 0.59969313],
            [0.27618746, 0.64322757],
            [0.54029566, 0.25492943],
            [0.30903526, 0.60166842],
            [0.2913432, 0.29636879],
            [0.78512072, 0.62340245],
            [0.29592116, 0.08400595],
            [0.87548394, 0.04877622],
            [0.21714791, 0.9607346],
            [0.92624074, 0.53441687],
            [0.53639253, 0.45127928],
            [0.99892031, 0.79537837],
            [0.84621631, 0.41891986],
            [0.39432819, 0.06768617],
            [0.92365693, 0.72217512],
            [0.95520914, 0.73956575],
            [0.820383, 0.53880139],
            [0.22378049, 0.9971974],
            [0.34023233, 0.91014706],
            [0.64960636, 0.35661133],
            [0.29976464, 0.33578931],
            [0.43202238, 0.11563227],
            [0.66764947, 0.52086962],
            [0.45431078, 0.94582745],
            [0.12819915, 0.33555344],
            [0.19287232, 0.8112075],
            [0.61214791, 0.71940626],
            [0.4522542, 0.47352186],
            [0.95623345, 0.74174186],
            [0.17340293, 0.89136853],
            [0.04600255, 0.53040724],
            [0.42493468, 0.41006649],
            [0.37631485, 0.88033853],
            [0.66951947, 0.29905739],
            [0.4151516, 0.77308712],
            [0.55762991, 0.26400156],
            [0.6280609, 0.53201974],
            [0.92727447, 0.61054975],
            [0.93206587, 0.42107549],
            [0.63885574, 0.37540613],
            [0.15303425, 0.57377797],
            [0.8208471, 0.16566631],
            [0.14889043, 0.35157346],
            [0.71724622, 0.57110725],
            [0.32866327, 0.8929578],
            [0.74435871, 0.47464421],
            [0.9252026, 0.21034329],
            [0.57039306, 0.54356078],
            [0.56611551, 0.02531317],
            [0.84830056, 0.01180542],
            [0.51282028, 0.73916524],
            [0.58795481, 0.46527371],
            [0.83259048, 0.98598188],
            [0.00242488, 0.83734691],
            [0.72505789, 0.04846931],
            [0.07312971, 0.30147979],
            [0.55250344, 0.23891255],
            [0.51161315, 0.46466442],
            [0.802125, 0.93440495],
            [0.9157825, 0.32441602],
            [0.44927665, 0.53380074],
            [0.67708372, 0.67527231],
            [0.81868924, 0.88356194],
            [0.48228814, 0.88668497],
            [0.39805433, 0.99341196],
            [0.86671752, 0.79016975],
            [0.01115417, 0.6924913],
            [0.34272199, 0.89543756],
            [0.40721675, 0.86164495],
            [0.26317679, 0.37334193],
            [0.74446787, 0.84782643],
            [0.55560143, 0.46405104],
            [0.73567977, 0.12776233],
            [0.28080322, 0.26036748],
            [0.17507419, 0.95540673],
            [0.54233783, 0.1196808],
            [0.76670967, 0.88396285],
            [0.61297539, 0.79057776],
            [0.9344029, 0.86252764],
            [0.48746839, 0.74942784],
            [0.18657635, 0.58127321],
            [0.10377802, 0.71463978],
            [0.7771771, 0.01463505],
            [0.7635042, 0.45498358],
            [0.83345861, 0.34749363],
            [0.38273809, 0.51890558],
            [0.33887574, 0.82842507],
            [0.02073685, 0.41776737],
            [0.68754547, 0.96430979],
            [0.4704215, 0.92717361],
            [0.72666234, 0.63241306],
            [0.48494401, 0.72003268],
            [0.52601215, 0.81641253],
            [0.71426732, 0.47077212],
            [0.00258906, 0.30377501],
            [0.35495269, 0.98585155],
            [0.65507544, 0.03458909],
            [0.10550588, 0.62032937],
            [0.60259145, 0.87110846],
            [0.04959159, 0.535785],
        ]
    )

    ell = c - 0.025
    h = c + 0.025

    r_box = np.array([[0.5, 0.5]])
    r_l = r_box - 0.5
    r_h = r_box + 0.5

    trees = AABoxes(ell, h)
    r_box = NegGeom(AABoxes(r_l, r_h))
    obstacles = UnionGeom([trees, r_box])

    start = np.zeros(2) + 0.05
    goal = np.array([0.95, 0.95])

    costs = [ConstObstacleCost(obstacles, cost=20.0), ConstCost(0.05)]
    cost_fn = AdditiveCosts(costs)
    return cost_fn, start, goal


def create_large_domain(force_start=False, force_goal=False, start_miss_cost=None, goal_miss_cost=None):
    """TODO"""
    cost_fn, start, goal = create_cost_large()

    n_points = 30
    traj = PointBSpline(dim=2, num_points=n_points)
    domain = RoverDomain(
        cost_fn,
        start=start,
        goal=goal,
        traj=traj,
        start_miss_cost=start_miss_cost,
        goal_miss_cost=goal_miss_cost,
        force_start=force_start,
        force_goal=force_goal,
        s_range=np.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    return domain


class Rover(BaseTestProblem):
    """
    Optimize trajectory from start point to end point over a rough terrain.
    Input is a 60D vector, reshaped to 30 2D points.

    Mainly optimize the integrating terrain roughness via splines + l1 norm between first 2 and last 2 coordinates to
    the start/end points.

    :param noise_std: Standard deviation of the Gaussian noise to be added to the function values.
    :parm num_cpus: Number of CPUs to use for parallel evaluation.
    """

    def __init__(
        self,
        noise_std: float,
    ) -> None:
        # domain of this function
        self.dim = 60
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        super().__init__(noise_std=noise_std, negate=False)

    @staticmethod
    def _l2cost(x, point):
        return 10 * np.linalg.norm(x - point, 1)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Rover benchmark.

        :param x: input configuration
        :return: function value
        """
        domain = create_large_domain(
            force_start=False, force_goal=False, start_miss_cost=self._l2cost, goal_miss_cost=self._l2cost
        )
        n_points = domain.traj.npoints

        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
        # maximum value of f
        f_max = 5.0
        f = ConstantOffsetFn(domain, f_max)
        f = NormalizedInputFn(f, raw_x_range)

        num_threads = torch.get_num_threads()

        if num_threads == 1 or X.shape[0] == 1:
            ret = [f(x) for x in X.detach().cpu().numpy()]
            return torch.tensor(ret).to(X)
        else:
            out = np.zeros(shape=X.shape[0])
            jobs = []
            X_np = X.detach().cpu().numpy()
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                for i in range(X.shape[0]):
                    x = X_np[i]
                    jobs.append(executor.submit(f, x))

            for i, j in enumerate(jobs):
                out[i] = j.result()

            ret = torch.from_numpy(out).to(X)
            return ret

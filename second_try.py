from theano import tensor as tt
from theano.ifelse import ifelse
from theano.compile.ops import as_op
import theano, time, numpy
theano.config.compute_test_value = "ignore"
a,b = tt.scalars('a', 'b')
x,y = tt.matrices('x', 'y')
import numpy as np
from scipy.integrate import odeint
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az 
# =============================================================================
# 
# =============================================================================
class FitzhughNagumoModel:
    def __init__(self, times, y0=None):
        self._y0 = np.array([-1, 1], dtype=np.float64)
        self._times = times

    def _simulate(self, parameters, times):
        a, b, c = [float(x) for x in parameters]

        def rhs(y, t, p):
            V, R = y
            dV_dt = (V - V ** 3 / 3 + R) * c
            dR_dt = (V - a + b * R) / -c
            return dV_dt, dR_dt

        values = odeint(rhs, self._y0, times, (parameters,), rtol=1e-3, atol=1e-3)
        return values

    def simulate(self, x):
        return self._simulate(x, self._times)
# =============================================================================
#     
# =============================================================================
n_states = 2
n_times = 200
true_params = [0.2, 0.2, 3.0]
noise_sigma = 0.5
FN_solver_times = np.linspace(0, 20, n_times)
ode_model = FitzhughNagumoModel(FN_solver_times)
sim_data = ode_model.simulate(true_params)
np.random.seed(42)
Y_sim = sim_data + np.random.randn(n_times, n_states) * noise_sigma
plt.figure(figsize=(15, 7.5))
plt.plot(FN_solver_times, sim_data[:, 0], color="darkblue", lw=4, label=r"$V(t)$")
plt.plot(FN_solver_times, sim_data[:, 1], color="darkgreen", lw=4, label=r"$R(t)$")
plt.plot(FN_solver_times, Y_sim[:, 0], "o", color="darkblue", ms=4.5, label="Noisy traces")
plt.plot(FN_solver_times, Y_sim[:, 1], "o", color="darkgreen", ms=4.5)
plt.legend(fontsize=15)
plt.xlabel("Time", fontsize=15)
plt.ylabel("Values", fontsize=15)
plt.title("Fitzhugh-Nagumo Action Potential Model", fontsize=25);
# =============================================================================
# 
# =============================================================================
# T.dvector dvector dmatrix
@as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector])
def th_forward_model(param1, param2, param3):
    param = [param1, param2, param3]
    th_states = ode_model.simulate(param)[:,0]

    return th_states

# =============================================================================
# 
# =============================================================================
# theano.config.exception_verbosity = "high"
draws = 1000
with pm.Model() as FN_model:

    a = pm.Gamma("a", alpha=2, beta=1)
    b = pm.Normal("b", mu=0, sigma=1)
    c = pm.Uniform("c", lower=0.1, upper=10)

    sigma = pm.HalfNormal("sigma", sigma=1)

    forward = th_forward_model(a, b, c)

    # cov = np.eye(2) * sigma ** 2

    # Y_obs = pm.MvNormal("Y_obs", mu=forward, cov=cov, observed=Y_sim)
    Y_obs = pm.Normal("Y_obs", mu=forward, sigma=sigma, observed=Y_sim[:,0])
    # startsmc = {v.name: np.random.uniform(1e-3, 2, size=draws) for v in FN_model.free_RVs}

    # trace_FN = pm.sample_smc(draws, start=startsmc)
    # Y_obs = pm.Normal("Y_obs", mu=forward, sigma=sigma, observed=Y)
    trace = pm.sample(1500,tune=1000, init="jitter+adapt_diag", cores=1)
    
    
   
    
az.plot_posterior(trace, kind="hist", bins=30, color="seagreen");    
    
    
    
    

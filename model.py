import numpy as np

def make_envelope_1d(s, lcl_phases, lcl_patterns, weights):
    """Returns a function of theta that computes the complex far-field envelope.
        s = distance between emitters, in units of wavelength
        lcl_phases = local phase shifts applied to each emitter 
        lcl_patterns = the envelope function defined by each individual emitter (array of functions)
        weights = array of weights, summing to one, describing how power is allocated across the array.
        """
    
    assert len(lcl_phases) == len(lcl_patterns) == len(weights)
    #number of emitters
    N = len(lcl_phases)
    # assuming they're indexed -M, ..., M
    assert N % 2 == 1
    M = N // 2

    def A(theta):
        phases = np.exp(-1j * 2 * np.pi * s * np.sin(theta) * np.arange(-M, M+1, 1))
        phases *= np.exp(1j * lcl_phases) * np.sqrt(weights)
        phases *= np.asarray([g(theta) for g in lcl_patterns])
        return np.sum(phases)
    
    return A

def power(env_fn, theta_vals):
    """Given an envelope function, returns the power at each of the specified angles"""
    return np.asarray([np.abs(env_fn(t))**2 for t in theta_vals])

def power_score(target_power, output_power):
    """Score of a given model power distribution, given a desired target. higher is better!"""
    return np.sum((target_power - output_power)**2)

def make_score_fn(s, fixed_phases, lcl_patterns, theta_vals):
    """ Returns a function which will compute a score given: 
        - target power distribution
        - power weights sampled from model
        - phases sampled from model
        """
    def score(model_phases, model_weights, target_power):
        env = make_envelope_1d(s, fixed_phases + model_phases, lcl_patterns, model_weights)
        return power_score(target_power, power(env, theta_vals))
    
    return score

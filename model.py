import numpy as np

def make_envelope_1(s, fixed_phases, lcl_phases, lcl_patterns, weights):
    """Returns a function of theta that computes the complex far-field envelope.
        s = distance between emitters, in units of wavelength
        lcl_phases = local phase shifts applied to each emitter 
        lcl_patterns = the envelope function defined by each individual emitter (array of functions)
        weights = array of weights, summing to one, describing how power is allocated across the array.
        """
    
    assert len(fixed_phases) == len(lcl_phases) == len(lcl_patterns) == len(weights)
    #number of emitters
    N = len(fixed_phases)
    # assuming they're indexed -M, ..., M
    assert N % 2 == 1
    M = N // 2

    def A(theta):
        phases = np.exp(-1j * 2 * np.pi * s * np.sin(theta) * np.arange(-M, M+1, 1))
        phases *= np.exp(1j * lcl_phases) * np.sqrt(weights)
        phases *= np.asarray([g(theta) for g in lcl_patterns])
        return np.sum(phases)
    
    return A


        
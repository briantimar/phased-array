import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

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

def power_loss(target_power, output_power):
    """Score of a given model power distribution, given a desired target. lower is better!"""
    target_power /= np.sum(target_power)
    output_power /= np.sum(output_power)

    return - np.sum(target_power * np.log(output_power))

def make_loss_fn(s, fixed_phases, lcl_patterns, theta_vals):
    """ Returns a function which will compute a score given: 
        - target power distribution
        - power weights sampled from model
        - phases sampled from model
        """
    def score(model_phases, model_weights, target_power):
        """ model_phases: (batch, N) array of phase values
            model_weights: (batch, N) array of emitter weights
            target_power: (batch, num_theta) array of target power values.
            returns: (batch,) tensor of loss values.
            """
        batch_size, N = model_phases.shape
        losses = []
        for i in range(batch_size):
            env = make_envelope_1d(s, fixed_phases + model_phases[i, :], lcl_patterns, 
                                        model_weights[i, :])

            losses.append(power_loss(target_power, power(env, theta_vals)))
        return np.asarray(losses)
    
    return score



class model1D(nn.Module):

    def __init__(self, input_length, output_length):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length

        assert input_length % 4 == 0

        self.output_dim = 256
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.linear = nn.Linear(self.input_length // 4, self.output_length)
        self.phase_head = nn.Linear(64, self.output_dim)
        self.wt_head = nn.Linear(64, self.output_dim)

        self.problayer = nn.LogSoftmax(dim=2)

    def forward(self, p):
        """p = (batchsize, array_length) array of power vals"""
        #input N,C,L
        y = self.conv1(p.unsqueeze(1)).relu()
        y = self.pool(y)
        y = self.conv2(y).relu()
        y = self.pool(y)
        # batchsize, length, hiddendim
        y = self.linear(y).permute(0, 2, 1)

        phase_logits = self.phase_head(y)
        wt_logits = self.wt_head(y)
 
        phase_dist = Categorical(logits= phase_logits)
        wt_dist = Categorical(logits=wt_logits)
        
        return phase_dist, wt_dist
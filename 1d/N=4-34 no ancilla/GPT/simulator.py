"""
Quantum State Simulator for ML Training Data Generation

This module simulates quantum measurements and generates training data for machine learning models
in quantum information tasks. It provides:

1. Quantum Information Measures:
   - bSqc: Batch quantum-classical entropy measure
   - Neg: Batch negativity measure for quantum entanglement
   - blogm: Batch matrix logarithm

2. QState Class - Quantum State Simulator:
   - Simulates quantum measurements in X, Y, Z bases
   - Generates measurement outcomes and shadow states for quantum shadow tomography
   - Supports both single-site and two-site measurements
   - Creates training data pairs (measurement_outcomes, target_quantum_states)

Primary Use: Generate synthetic quantum measurement data for training ML models to:
- Reconstruct quantum states from measurement statistics
- Predict quantum properties from measurement data

The simulator is particularly useful for 1D quantum systems without ancilla qubits,
where direct quantum state reconstruction from measurement data is the main task.
"""

import torch

dtype = torch.complex128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def blogm(A):
    E, U = torch.linalg.eig(A)
    # E += 1e-10
    logE = torch.log(E.abs()).to(U.dtype)
    logA = torch.bmm(torch.bmm(U, torch.diag_embed(logE, offset=0, dim1=-2, dim2=-1)), U.conj().mT)
    return logA

def bSqc(rhoQ, rhoC):
    return -torch.vmap(torch.trace)(rhoQ@blogm(rhoC)).real

def Neg(rhoS, rhoC):
    rhoC_pt = rhoC.view(-1,2,2,2,2).permute(0,1,4,3,2).reshape(-1,4,4)
    rhoS_pt = rhoS.view(-1,2,2,2,2).permute(0,1,4,3,2).reshape(-1,4,4)
    e, v = torch.linalg.eig(rhoC_pt)
    # e += 1e-10
    mask = e.real < 0
    negative_v = v * mask.unsqueeze(1)
    P = torch.bmm(negative_v, negative_v.mT.conj()) # projection matrix
    return -torch.vmap(torch.trace)(torch.bmm(P, rhoS_pt)).real

class QState(torch.nn.Module):
    
    
    def __init__(self, state):
        super().__init__()
        self._state = state
        self.N = state.dim()
        pauli = torch.tensor([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]], device=device, dtype=dtype)
        basis = torch.linalg.eig(pauli)[1][1:].mT # (3, 2, 2)
        C4 = torch.tensor([[[1/torch.tensor(2).sqrt(), 1/torch.tensor(2).sqrt()],
                            [1/torch.tensor(2).sqrt(), -1/torch.tensor(2).sqrt()]],
                   
                            [[-1j/torch.tensor(2).sqrt(), 1/torch.tensor(2).sqrt()],
                            [1/torch.tensor(2).sqrt(), -1j/torch.tensor(2).sqrt()]],
                   
                            [[1, 0],
                            [0, 1]]], device=device, dtype=dtype) # (3, 2, 2)
        # Registering the tensors as buffers
        self.register_buffer('pauli', pauli)
        self.register_buffer('basis', basis)
        self.register_buffer('C4', C4)
        
        
    # bosseq: 0,1,2 -> X,Y,Z (batch, N)
    # outseq: 0,1 -> +,-
    def measure(self, obsseq):
        state = self._state.clone()
        shape = (obsseq.shape[0],) + state.shape
        state = state[None,...].expand(shape) # make copies of state
        # act C4 rotations to each state
        for i in range(self.N):
            state = torch.swapaxes(state, 1, i+1) # put target site to the first place
            U = self.C4[obsseq[:,i]] # (batch, 2, 2)
            state = torch.bmm(U, state.reshape(obsseq.shape[0], 2, -1)).reshape(shape)
            state = torch.swapaxes(state, 1, i+1) # put target site back
        # sample for outcomes
        probs = (state.conj() * state).reshape(obsseq.shape[0], -1).real # (batch, 2^N)
        idx = torch.multinomial(probs, num_samples=1) # (batch, )
        mask = 2**torch.arange(self.N - 1, -1, -1, device=device, dtype=idx.dtype)
        outseq = (idx & mask) != 0 # (batch, N)
        return outseq.int()
    
    # # input_seq: 0,1 -> +,-
    # def get_data(self, batch, site=None):
    #     # if site is specified, then leave it for random XYZ measure
    #     if site is None:
    #         site_idx = torch.randint(self.N, (batch, ), device=device) # (batch, ) randomly select one site for measuring XYZ
    #     else:
    #         site_idx = torch.full((batch, ), site, device=device) # (batch, )
    #     # create obsseq
    #     obsseq = torch.full((batch, self.N), 2, device=device)
    #     site_obs = torch.randint(3, (batch, ), device=device)
    #     obsseq = obsseq.scatter(1, site_idx[...,None], site_obs[...,None])
    #     # measure state with obsseq
    #     outseq = self.measure(obsseq) # (batch, N)
    #     # construct post-measure state
    #     out = torch.gather(outseq, 1, site_idx[...,None])
    #     obs_basis = self.basis[site_obs] # (batch, 2, 2)
    #     shadow_state = obs_basis.gather(1, out.to(torch.int64).unsqueeze(1).expand(-1, -1, 2)).squeeze(1) # (batch, 2)
    #     outseq0 = outseq.scatter(1, site_idx[...,None], torch.zeros_like(site_idx[...,None]).int()).unsqueeze(-2)
    #     outseq1 = outseq.scatter(1, site_idx[...,None], torch.ones_like(site_idx[...,None]).int()).unsqueeze(-2)
    #     data_seq = torch.cat([outseq0, outseq1], -2)
    #     #obs_seq = torch.cat([site_obs.view(-1, 1), out.view(-1, 1)], -1) # (batch, 2)
    #     return data_seq, shadow_state#, obs_seq
    
    # input_seq: -1,1,0 -> +,-,shadow
    def get_data(self, batch, site=None, Z=False):
        # if site is specified, then leave it for random XYZ measure
        if site is None:
            site_idx = torch.randint(self.N, (batch, ), device=device) # (batch, ) randomly select one site for measuring XYZ
        else:
            site_idx = torch.full((batch, ), site, device=device) # (batch, )
        # create obsseq
        obsseq = torch.full((batch, self.N), 2, device=device)
        site_obs = torch.randint(2+Z, (batch, ), device=device)
        obsseq = obsseq.scatter(1, site_idx[...,None], site_obs[...,None])
        # measure state with obsseq
        outseq = self.measure(obsseq) # (batch, N)
        # construct post-measure state
        out = torch.gather(outseq, 1, site_idx[...,None])
        obs_basis = self.basis[site_obs] # (batch, 2, 2)
        shadow_state = obs_basis.gather(1, out.to(torch.int64).unsqueeze(1).expand(-1, -1, 2)).squeeze(1) # (batch, 2)
        outseq = (outseq * 2) - 1 # 0,1 -> -1,1 encoding
        data_seq = outseq.scatter(1, site_idx[...,None], torch.zeros_like(site_idx[...,None]).int())
        #obs_seq = torch.cat([site_obs.view(-1, 1), out.view(-1, 1)], -1) # (batch, 2)
        return data_seq, shadow_state#, obs_seq
    
        # input_seq: -1,1,0 -> +,-,shadow
    def data2(self, batch, Z=False):
        # if site is specified, then leave it for random XYZ measure
        site_idx = torch.tensor([[self.N-2, self.N-1]]).expand(batch, -1)
        # create obsseq
        # obsseq = torch.randint(3, (batch, self.N), device=device)
        # insert a Z randomly
        obsseq = torch.full((batch, self.N), 0, device=device) # ALL X
        Z_idx = torch.randint(self.N-2, (batch, 1))
        Z_obs = torch.full((batch, 1), 2, device=device)
        obsseq = obsseq.scatter(1, Z_idx, Z_obs)
        
        site_obs = torch.randint(2+Z, (batch, 2), device=device)
        obsseq = obsseq.scatter(1, site_idx, site_obs)
        # measure state with obsseq
        outseq = self.measure(obsseq) # (batch, N)
        # construct post-measure state
        out = outseq[:,-2:] # (batch, 2) last 2 outcomes
        obs_basis0 = self.basis[site_obs[:,0]] # (batch, 2, 2)
        shadow_state0 = obs_basis0.gather(1, out[:,0].to(torch.int64).view(-1, 1, 1).expand(-1, -1, 2)).squeeze(1) # (batch, 2)
        obs_basis1 = self.basis[site_obs[:,1]] # (batch, 2, 2)
        shadow_state1 = obs_basis1.gather(1, out[:,1].to(torch.int64).view(-1, 1, 1).expand(-1, -1, 2)).squeeze(1) # (batch, 2)
        shadow_state = torch.vmap(torch.kron)(shadow_state0, shadow_state1) # (batch, 4)
        # construct rhoS
        I = torch.eye(2, 2, device=device)[None,...].expand(batch, -1, -1)
        rhoS0 = 3*torch.vmap(torch.outer)(shadow_state0.conj(), shadow_state0) - I
        rhoS1 = 3*torch.vmap(torch.outer)(shadow_state1.conj(), shadow_state1) - I
        rhoS = torch.vmap(torch.kron)(rhoS0, rhoS1)
        return (outseq+2*obsseq)[:,:-2], shadow_state, rhoS
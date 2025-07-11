import pandas
import torch

dtype = torch.complex128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pauli = torch.tensor([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]], device=device, dtype=dtype)
basis = torch.linalg.eig(pauli)[1][1:].mT # (3, 2, 2)

def torch_data(filename, shuffle=False):
    out = pandas.read_pickle(filename)
    out = [torch.tensor(d['m']) for d in out]
    data = {}
    s = 0
    for i in range(3):
        for j in range(3):
            data[(i,j)] = (out[s][:,1:][:,:-1], torch.cat([out[s][:,-1:], out[s][:,:1]], -1))
            s += 1
    prepseq, shadow_state, rhoS = [], [], []
    for k in data.keys():
        obsseq = torch.tensor(k)
        # construct post-measure state
        probseq = data[k][1].to(dtype=torch.int64).to(device=device) # (repetition, 2) last 2 outcomes
        obs_basis0 = basis[k[0]].unsqueeze(0).expand(probseq.shape[0], -1, -1) # (repetition, 2, 2)
        shadow_state0 = obs_basis0.gather(1, probseq[:,0].view(-1, 1, 1).expand(-1, -1, 2)).squeeze(1) # (repetition, 2)
        obs_basis1 = basis[k[1]].unsqueeze(0).expand(probseq.shape[0], -1, -1) # (repetition, 2, 2)
        shadow_state1 = obs_basis1.gather(1, probseq[:,1].view(-1, 1, 1).expand(-1, -1, 2)).squeeze(1) # (repetition, 2)
        shadow_state01 = torch.vmap(torch.kron)(shadow_state0, shadow_state1) # (batch, 4)
        # construct rhoS
        I = torch.eye(2, 2, device=device)[None,...].expand(shadow_state01.shape[0], -1, -1)
        rhoS0 = 3*torch.vmap(torch.outer)(shadow_state0.conj(), shadow_state0) - I
        rhoS1 = 3*torch.vmap(torch.outer)(shadow_state1.conj(), shadow_state1) - I
        rhoS01 = torch.vmap(torch.kron)(rhoS0, rhoS1)
        # collect result
        prepseq.append(data[k][0].to(dtype=torch.int64).to(device=device))
        shadow_state.append(shadow_state01)
        rhoS.append(rhoS01)
    prepseq = torch.cat(prepseq, 0).to(torch.int64)
    shadow_state = torch.cat(shadow_state, 0)
    rhoS = torch.cat(rhoS, 0)
    if shuffle:
        indices = torch.randperm(prepseq.shape[0])
        prepseq = prepseq[indices]
        shadow_state = shadow_state[indices]
        rhoS = rhoS[indices]
    #print(prepseq.shape)
    return prepseq, shadow_state, rhoS
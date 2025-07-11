import torch

dtype = torch.complex128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class tel_mpo(torch.nn.Module):
#     def __init__(self, L, bond=8):
#         super().__init__()
#         self.L = L
#         self.bond = bond
#         preps = torch.randn(L-2, 2, bond, bond ,dtype=dtype, device=device)
#         probs = torch.randn(2, bond, 2, bond, dtype=dtype, device=device)
#         self.preps = torch.nn.Parameter(preps) # (site, 0/1, bond, bond)
#         self.probs = torch.nn.Parameter(probs) # (l/r, bond, phy, bond)
        
#     def forward(self, measure):
#         psi = self.probs[0].unsqueeze(0).expand(measure.shape[0], -1, -1, -1).flatten(1,2) # (batch, bond*phy, bond)
#         for i in range(measure.shape[-1]):#range(self.L-2):
#             prep = self.preps[i][measure[:,i]] # (batch, bond, bond)
#             psi = torch.bmm(psi, prep)
#         psi = torch.bmm(psi, self.probs[1].unsqueeze(0).expand(measure.shape[0], -1, -1, -1).flatten(2,3)) # (batch, bond*phy, bond*phy)
#         psi = psi.view(-1, self.bond, 4, self.bond).permute(0, 2, 1, 3).flatten(2, 3) # (batch, 4, bond*bond)
#         rho = torch.bmm(psi, psi.mT.conj())
#         rho /= torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
#         return rho
    
class tel_mpo(torch.nn.Module):
    def __init__(self, L, bond=8):
        super().__init__()
        self.L = L
        self.bond = bond
        blocks = torch.randn(L, 2, bond, bond ,dtype=dtype, device=device)
        self.blocks = torch.nn.Parameter(blocks) # (site, 0/1, bond, bond)
        
    def forward(self, measure):
        psi = self.blocks[0].permute(1,0,2).unsqueeze(0).expand(measure.shape[0], -1, -1, -1).flatten(1,2) # (batch, bond*phy, bond)
        for i in range(measure.shape[-1]):#range(self.L-2):
            prep = self.blocks[i+1][measure[:,i]] # (batch, bond, bond)
            psi = torch.bmm(psi, prep)
        psi = torch.bmm(psi, self.blocks[measure.shape[-1]+1].permute(1,0,2).unsqueeze(0).expand(measure.shape[0], -1, -1, -1).flatten(2,3)) # (batch, bond*phy, bond*phy)
        psi = psi.view(-1, self.bond, 4, self.bond).permute(0, 2, 1, 3).flatten(2, 3) # (batch, 4, bond*bond)
        rho = torch.bmm(psi, psi.mT.conj())
        rho /= torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
        return rho
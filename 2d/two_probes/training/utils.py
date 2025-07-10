import pickle
import torch
import os
import json

dtype = torch.complex128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pauli = torch.tensor([[[1,0],[0,1]],[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]], device=device, dtype=dtype)
basis = torch.linalg.eig(pauli)[1][1:].mT # (3, 2, 2)

def eps(rho, e=0.1):
    I = torch.eye(rho.shape[-1], rho.shape[-1], dtype=rho.dtype, device=rho.device)[None,...].expand(rho.shape[0], -1, -1)/rho.shape[-1]
    return (1-e)*rho + e*I

def purity(rhoQ, rhoC):
    return 2*torch.vmap(torch.trace)(rhoQ@rhoC).real - torch.vmap(torch.trace)(rhoC@rhoC).real

def blogm(A):
    E, U = torch.linalg.eig(A)
    #E += 1e-5
    logE = torch.log(E.abs()).to(U.dtype)
    logA = torch.bmm(torch.bmm(U, torch.diag_embed(logE, offset=0, dim1=-2, dim2=-1)), U.conj().mT)
    return logA

def bSqc(rhoQ, rhoC):
    return -torch.vmap(torch.trace)(rhoQ@blogm(rhoC)).real
    
def Neg(rhoS, rhoC):
    rhoC_pt = rhoC.view(-1,2,2,2,2).permute(0,1,4,3,2).reshape(-1,4,4)
    rhoS_pt = rhoS.view(-1,2,2,2,2).permute(0,1,4,3,2).reshape(-1,4,4)
    e, v = torch.linalg.eig(rhoC_pt)
    #e += 1e-5
    mask = e.real < 0
    negative_v = v * mask.unsqueeze(1)
    P = torch.bmm(negative_v, negative_v.mT.conj()) # projection matrix
    return -torch.vmap(torch.trace)(torch.bmm(P, rhoS_pt)).real

def Sa(rhoS, rhoC):
    rhoCa = torch.einsum('bijkj->bik', rhoC.view(-1,2,2,2,2))
    rhoSa = torch.einsum('bijkj->bik', rhoS.view(-1,2,2,2,2))
    return bSqc(rhoSa, rhoCa)

def torch_data(filename, L, theta_idx):
    out = []
    with (open(f'data/{filename}', "rb")) as openfile:
        while True:
            try:
                out.append(pickle.load(openfile))
            except EOFError:
                break
    data = {}
    s = 0
    for i in range(3):
        for j in range(3):
            m = torch.cat([torch.tensor(out[s][k][theta_idx]['m']) for k in range(10)])
            data[(i,j)] = (torch.cat([m[:,1:][:,:L-2], m[:,L:]], -1), torch.cat([m[:,0][...,None], m[:,L-1][...,None]], -1)) # (prep, probe)
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
        rhoS0 = 3*torch.vmap(torch.outer)(shadow_state0, shadow_state0.conj()) - I
        rhoS1 = 3*torch.vmap(torch.outer)(shadow_state1, shadow_state1.conj()) - I
        rhoS01 = torch.vmap(torch.kron)(rhoS0, rhoS1)
        # collect result
        prepseq.append(data[k][0].to(dtype=torch.int64).to(device=device))
        shadow_state.append(shadow_state01)
        rhoS.append(rhoS01)
    prepseq = torch.cat(prepseq, 0).to(torch.int64)
    shadow_state = torch.cat(shadow_state, 0)
    rhoS = torch.cat(rhoS, 0)
    return prepseq, shadow_state, rhoS

def shuffle(prepseq, shadow_state, rhoS):
    indices = torch.randperm(prepseq.shape[0])
    prepseq = prepseq[indices]
    shadow_state = shadow_state[indices]
    rhoS = rhoS[indices]
    return prepseq, shadow_state, rhoS

# ============= TRAIN/TEST SPLIT UTILITIES =============

def create_train_test_split(prepseq_all, shadow_all, rhoS_all, train_size, test_size, batch_size):
    """
    Create non-overlapping train/test split with batching.
    
    Args:
        prepseq_all: Full preparation sequence tensor
        shadow_all: Full shadow state tensor  
        rhoS_all: Full rhoS tensor
        train_size: Number of samples for training
        test_size: Number of samples for testing
        batch_size: Batch size for reshaping data
    
    Returns:
        tuple: (train_data, test_data) where each is a dict containing:
               - 'prepseq': batched preparation sequences
               - 'shadow_state': batched shadow states
               - 'rhoS': batched rhoS tensors
               - 'indices': original indices used
    """
    # Ensure we have enough data
    total_needed = train_size + test_size
    assert total_needed <= prepseq_all.shape[0], f'Not enough data: need {total_needed}, have {prepseq_all.shape[0]}'
    
    # Create non-overlapping train/test split with explicit indices
    test_indices = torch.arange(0, test_size)
    train_indices = torch.arange(test_size, test_size + train_size)
    
    # Verify no overlap (explicit check)
    assert len(set(test_indices.tolist()) & set(train_indices.tolist())) == 0, "Train/test indices overlap!"
    
    # Split data using indices
    prepseq_test = prepseq_all[test_indices]
    shadow_state_test = shadow_all[test_indices] 
    rhoS_test = rhoS_all[test_indices]
    
    prepseq_train = prepseq_all[train_indices]
    shadow_state_train = shadow_all[train_indices]
    rhoS_train = rhoS_all[train_indices]
    
    print(f'test size={prepseq_test.shape[0]}, train size={prepseq_train.shape[0]}')
    print(f'test indices: [{test_indices[0]}-{test_indices[-1]}], train indices: [{train_indices[0]}-{train_indices[-1]}]')
    
    # Split in batches
    seq_len = prepseq_all.shape[-1]  # Read sequence length from actual data
    prepseq_train_batched = prepseq_train.view(-1, batch_size, seq_len)
    shadow_state_train_batched = shadow_state_train.view(-1, batch_size, 4)
    rhoS_train_batched = rhoS_train.view(-1, batch_size, 4, 4)
    
    prepseq_test_batched = prepseq_test.view(-1, batch_size, seq_len)
    shadow_state_test_batched = shadow_state_test.view(-1, batch_size, 4)
    rhoS_test_batched = rhoS_test.view(-1, batch_size, 4, 4)
    
    # Prepare return data
    train_data = {
        'prepseq': prepseq_train_batched,
        'shadow_state': shadow_state_train_batched,
        'rhoS': rhoS_train_batched,
        'indices': train_indices
    }
    
    test_data = {
        'prepseq': prepseq_test_batched,
        'shadow_state': shadow_state_test_batched,
        'rhoS': rhoS_test_batched,
        'indices': test_indices
    }
    
    return train_data, test_data

# ============= CHECKPOINT UTILITIES =============

def save_checkpoint(model, optimizer, epoch, checkpoint_num=0, save_dir='checkpoints', filename_prefix='checkpoint'):
    """
    Save model checkpoint with epoch and checkpoint number naming.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        checkpoint_num: Checkpoint number within the epoch (default: 0)
        save_dir: Directory to save checkpoints (default: 'checkpoints')
        filename_prefix: Prefix for checkpoint filename (default: 'checkpoint')
    
    Returns:
        str: Full path of saved checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate checkpoint filename: checkpoint_epoch{epoch}_step{checkpoint_num}.pt
    checkpoint_name = f"{filename_prefix}_epoch{epoch:04d}_step{checkpoint_num:04d}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'checkpoint_num': checkpoint_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, optimizer, epoch, checkpoint_num=0, save_dir='checkpoints', filename_prefix='checkpoint'):
    """
    Load model checkpoint by epoch and checkpoint number.
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into  
        epoch: Epoch number of checkpoint to load
        checkpoint_num: Checkpoint number within the epoch (default: 0)
        save_dir: Directory containing checkpoints (default: 'checkpoints')
        filename_prefix: Prefix for checkpoint filename (default: 'checkpoint')
    
    Returns:
        dict: Checkpoint data including epoch, checkpoint_num, path
    """
    # Generate checkpoint filename
    checkpoint_name = f"{filename_prefix}_epoch{epoch:04d}_step{checkpoint_num:04d}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    return {
        'epoch': checkpoint['epoch'],
        'checkpoint_num': checkpoint['checkpoint_num'], 
        'path': checkpoint_path
    }


def save_checkpoint_and_test(model, optimizer, epoch, checkpoint_num, 
                           temp_train, temp_test_test,
                           l_train, l_test, 
                           prepseq_train, shadow_state_train, rhoS_train,
                           prepseq_test, shadow_state_test, rhoS_test,
                           device, save_dir, filename_prefix, d, theta_idx, num_check,
                           is_final=False):
    """
    Helper function to save checkpoint and run test loop
    
    Args:
        model: The model to save and test
        optimizer: The optimizer to save
        epoch: Current epoch number
        checkpoint_num: Checkpoint number within the epoch
        temp_train: Temporary training metrics storage
        temp_test_test: Temporary test metrics storage for test data
        l_train: Training metrics lists
        l_test: Test metrics lists
        prepseq_train: Training preparation sequences (unused, kept for compatibility)
        shadow_state_train: Training shadow states (unused, kept for compatibility)
        rhoS_train: Training rhoS tensors (unused, kept for compatibility)
        prepseq_test: Test preparation sequences (should be preprocessed before calling this function)
        shadow_state_test: Test shadow states
        rhoS_test: Test rhoS tensors
        device: Device to use for computations
        save_dir: Directory to save checkpoints
        filename_prefix: Prefix for checkpoint filename
        d: Model dimension parameter
        theta_idx: Theta index parameter
        num_check: Number of checkpoints per epoch
        is_final: Whether this is the final checkpoint of an epoch
    """
    # Calculate and store mean training metrics
    l_train['loss'].append(torch.tensor(temp_train['loss']).mean().item())
    l_train['msk on Sqc'].append(torch.tensor(temp_train['msk on Sqc']).mean().item())
    
    # Run test loop
    with torch.no_grad():
        checkpoint_type = "final" if is_final else f"{checkpoint_num}/{num_check}"
        print(f'Running test at epoch {epoch}, checkpoint {checkpoint_type}')
        model.eval()
        
        # Test on test data (mask off)
        test_batches = prepseq_test.shape[0]
        for j in range(test_batches):
            prepseq_batch, shadow_state_batch, rhoS_batch = prepseq_test[j].clone(), shadow_state_test[j].clone(), rhoS_test[j].clone()
            prepseq_batch = prepseq_batch.to(device)
            shadow_state_batch = shadow_state_batch.to(device)
            rhoS_batch = rhoS_batch.to(device)
            rhoC = model(prepseq_batch, False)
            temp_test_test['msk off Sqc'].extend(bSqc(rhoS_batch, rhoC).tolist())
            temp_test_test['msk off Neg'].extend(Neg(rhoS_batch, rhoC).tolist())
            temp_test_test['msk off Sa'].extend(Sa(rhoS_batch, rhoC).tolist())

        
        # Calculate and store mean test metrics
        l_test['msk off Sqc'].append(torch.tensor(temp_test_test['msk off Sqc']).mean().item())
        l_test['msk off Neg'].append(torch.tensor(temp_test_test['msk off Neg']).mean().item())
        l_test['msk off Sa'].append(torch.tensor(temp_test_test['msk off Sa']).mean().item())
        
        model.train()  # Back to training mode
    
    # Save checkpoint
    save_checkpoint(model, optimizer, epoch, checkpoint_num, 
                  save_dir=save_dir, 
                  filename_prefix=filename_prefix)
    
    # Clear temporary storage
    temp_train['msk on Sqc'].clear()
    temp_train['loss'].clear()
    temp_test_test['msk off Sqc'].clear()
    temp_test_test['msk off Neg'].clear()
    temp_test_test['msk off Sa'].clear()
    
    # Print current metrics
    checkpoint_type_str = "final checkpoint" if is_final else "checkpoint"
    print('epoch: %3d | %s: %3d | d: %3d | theta_idx: %3d | train Sqc: %.4f | loss: %.4f | test Sqc: %.4f | test Neg: %.4f | test Sa: %.4f' 
          %(epoch, checkpoint_type_str, checkpoint_num, d, theta_idx, l_train['msk on Sqc'][-1], l_train['loss'][-1], l_test['msk off Sqc'][-1], l_test['msk off Neg'][-1], l_test['msk off Sa'][-1]))

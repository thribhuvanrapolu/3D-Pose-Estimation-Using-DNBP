import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math

def _get_grid_dims(n):
    """Calculates the optimal number of rows and columns for a subplot grid."""
    if n <= 0:
        return 1, 1
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / float(cols)))
    return rows, cols

def _calculate_density_from_particles(grid, particles, weights, bandwidth=0.05):
    """
    Estimates a density distribution on a grid from weighted particles.
    This is a generalized version of your 'density_estimation' function.

    Args:
        grid (Tensor): The grid to evaluate the density on. Shape [H, W, 2].
        particles (Tensor): The particle locations. Shape [N_particles, 2].
        weights (Tensor): The weight for each particle. Shape [N_particles].
        bandwidth (float): The bandwidth of the Gaussian kernel.

    Returns:
        Tensor: The estimated density on the grid.
    """
    # Prepare tensors for broadcasting
    # Unsqueeze to add batch/channel dims for broadcasting
    grid = grid.unsqueeze(0).unsqueeze(0) # [1, 1, H, W, 2]
    particles = particles.view(1, -1, 2).unsqueeze(1).unsqueeze(1) # [1, 1, 1, 1, N_particles, 2]
    weights = weights.view(1, -1).unsqueeze(1).unsqueeze(1) # [1, 1, 1, N_particles]

    # Use double for precision
    grid, particles, weights = grid.double(), particles.double(), weights.double()

    # Gaussian Kernel Density Estimation
    diff_sq = (((grid - particles) / bandwidth) ** 2).sum(dim=-1)
    exp_val = torch.exp(-0.5 * diff_sq)
    factor = 1.0 / (bandwidth * np.sqrt(2 * np.pi)) # Normalization factor

    # Sum the weighted kernel values
    out = (weights * factor * exp_val).sum(dim=-1)
    return out.squeeze()


def plot_unaries(bpnet, grid, x, unnormed=None, std=None, tru=None, to_show=0, est_bounds=1.0, nump=100, fname=None):
    """Plots the unary likelihoods for all nodes on a grid."""
    rows, cols = _get_grid_dims(bpnet.num_nodes)
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    ax = ax.flatten()

    colors = plt.cm.get_cmap('viridis', bpnet.num_nodes)
    if unnormed is None: unnormed = x

    for i in range(bpnet.num_nodes):
        current_ax = ax[i]
        current_ax.set_title(f'Unary Node {i}')
        current_ax.set_xlim(-est_bounds, est_bounds)
        current_ax.set_ylim(-est_bounds, est_bounds)
        current_ax.imshow(unnormed[to_show].cpu().permute(1, 2, 0), extent=[-est_bounds, est_bounds, -est_bounds, est_bounds])

        if std:
            out = bpnet.likelihood(i, x[to_show].unsqueeze(0), std, tru[to_show, i], grid.unsqueeze(0)).cpu()
        else:
            out = bpnet.likelihood(i, x[to_show].unsqueeze(0), grid.unsqueeze(0)).cpu()

        out = F.interpolate(out.view(1, 1, nump, nump).flip(2), (500, 500), mode='bilinear').squeeze()
        out = (out / out.max()).detach().numpy()

        color_map = colors(i / (bpnet.num_nodes - 1)) if bpnet.num_nodes > 1 else colors(0.5)
        out_c = plt.cm.colors.LinearSegmentedColormap.from_list('custom', [(0,0,0,0), color_map])(out)

        current_ax.imshow(out_c, extent=[-est_bounds, est_bounds, -est_bounds, est_bounds])

    for i in range(bpnet.num_nodes, len(ax)): ax[i].set_visible(False)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=fig.dpi); plt.close('all')
    else:
        plt.show()

def plot_belief_particles(bpnet, grid, x, unnormed=None, with_particles=True, std=None, to_show=0, est_bounds=1.0, nump=100, s=2, fname=None):
    """Plots the final belief (particle densities and locations) for all nodes."""
    rows, cols = _get_grid_dims(bpnet.num_nodes)
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    ax = ax.flatten()

    colors = plt.cm.get_cmap('hsv', bpnet.num_nodes)
    if unnormed is None: unnormed = x

    for i in range(bpnet.num_nodes):
        current_ax = ax[i]
        current_ax.set_title(f'Belief Node {i}')
        current_ax.imshow(unnormed[to_show].cpu().permute(1, 2, 0), extent=[-est_bounds, est_bounds, -est_bounds, est_bounds])

        if with_particles:
            particles = bpnet.belief_particles[i].view(bpnet.batch_size, -1, bpnet.particle_size)[to_show]
            current_ax.scatter(x=particles[:, 0].cpu().detach().numpy(),
                               y=particles[:, 1].cpu().detach().numpy(),
                               c=[colors(i)], s=s)
        if std:
            out = bpnet.density_estimation(i, grid, std=std).cpu()[to_show]
        else:
            out = bpnet.density_estimation(i, grid).cpu()[to_show]

        out = F.interpolate(out.view(1, 1, nump, nump).flip(2), (500, 500), mode='bilinear').squeeze()
        out = (out / out.max()).detach().numpy()
        
        out_c = plt.cm.colors.LinearSegmentedColormap.from_list('custom', [(0,0,0,0), colors(i)])(out)
        current_ax.imshow(out_c, extent=[-est_bounds, est_bounds, -est_bounds, est_bounds])

    for i in range(bpnet.num_nodes, len(ax)): ax[i].set_visible(False)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=fig.dpi); plt.close('all')
    else:
        plt.show()

def plot_pairwise_sampling(bpnet, num_samples=10000, est_bounds=1.0, s=5, fname=None):
    """Plots samples from the pairwise potentials for all edges."""
    rows, cols = _get_grid_dims(bpnet.num_edges)
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    ax = ax.flatten()

    for i in range(bpnet.num_edges):
        current_ax = ax[i]
        # Assumes bpnet.edges exists to provide context, e.g., [(0, 1), (1, 2), ...]
        if hasattr(bpnet, 'edges'):
            current_ax.set_title(f'Edge: {bpnet.edges[i][0]}-{bpnet.edges[i][1]}')
        else:
            current_ax.set_title(f'Edge {i}')

        current_ax.set_ylim(-est_bounds, est_bounds)
        current_ax.set_xlim(-est_bounds, est_bounds)
        current_ax.set_aspect('equal', adjustable='box')

        if bpnet.multi_edge_samplers: # Handles the special case from original code
            samp = bpnet.edge_samplers[i][0](num_samples).cpu().detach()
        else:
            samp = bpnet.edge_samplers[i](num_samples).cpu().detach()

        current_ax.scatter(x=samp[:, 0], y=samp[:, 1], c='r', s=s, alpha=0.3)

    for i in range(bpnet.num_edges, len(ax)): ax[i].set_visible(False)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=fig.dpi); plt.close('all')
    else:
        plt.show()

def plot_pairwise_densities(bpnet, grid, std=None, est_bounds=1.0, nump=100, fname=None):
    """Plots the density of the pairwise potentials for all edges."""
    rows, cols = _get_grid_dims(bpnet.num_edges)
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    ax = ax.flatten()

    for i in range(bpnet.num_edges):
        current_ax = ax[i]
        if hasattr(bpnet, 'edges'):
            current_ax.set_title(f'Edge: {bpnet.edges[i][0]}-{bpnet.edges[i][1]}')
        else:
            current_ax.set_title(f'Edge {i}')
            
        current_ax.set_ylim(-est_bounds, est_bounds)
        current_ax.set_xlim(-est_bounds, est_bounds)

        if std:
            out = bpnet.edge_densities[i](grid.squeeze(1), std).cpu()
        else:
            out = bpnet.edge_densities[i](grid.squeeze(1)).cpu()

        out = F.interpolate(out.view(1, 1, nump, nump).flip(2), (500, 500), mode='bilinear').squeeze()
        current_ax.imshow(out.detach().numpy(), extent=[-est_bounds, est_bounds, -est_bounds, est_bounds], cmap='viridis')

    for i in range(bpnet.num_edges, len(ax)): ax[i].set_visible(False)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=fig.dpi); plt.close('all')
    else:
        plt.show()

def plot_timedelta_sampling(bpnet, num_samples=10000, est_bounds=1.0, s=5, ground_truth=False, fname=None):
    """Plots samples from the time-delta potentials (if applicable)."""
    rows, cols = _get_grid_dims(bpnet.num_nodes)
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    ax = ax.flatten()

    colors = plt.cm.get_cmap('hsv', bpnet.num_nodes)

    for i in range(bpnet.num_nodes):
        current_ax = ax[i]
        current_ax.set_title(f'Time Delta Node {i}')
        current_ax.set_ylim(-est_bounds, est_bounds)
        current_ax.set_xlim(-est_bounds, est_bounds)
        current_ax.set_aspect('equal', adjustable='box')

        if ground_truth:
            samp = bpnet.time_samplers[i](i, num_samples).cpu().detach()
        else:
            samp = bpnet.time_samplers[i](num_samples).cpu().detach()
        current_ax.scatter(x=samp[:, 0], y=samp[:, 1], c=[colors(i)], s=s)

    for i in range(bpnet.num_nodes, len(ax)): ax[i].set_visible(False)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=fig.dpi); plt.close('all')
    else:
        plt.show()

def plot_msg_wgts(bpnet, x, mode, unnormed=None, to_show=0, bin_size=0.2, fname=None):
    """Plots a 2D histogram of message particle weights for all nodes."""
    rows, cols = _get_grid_dims(bpnet.num_nodes)
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    ax = ax.flatten()

    colors = plt.cm.get_cmap('hsv', bpnet.num_nodes)
    if unnormed is None: unnormed = x

    if mode == 'w_lik':
        wgt_to_use = [w[to_show].cpu() for w in bpnet.belief_weights_lik]
    elif mode == 'w_unary':
        wgt_to_use = [w[to_show].cpu() for w in bpnet.message_weights_unary]
    elif mode == 'w_neigh':
        wgt_to_use = [w[to_show].cpu() for w in bpnet.message_weights_neigh]
    else:
        raise ValueError("mode must be one of 'w_lik', 'w_unary', or 'w_neigh'")
    msgs = [m[to_show].cpu() for m in bpnet.message_particles]

    for i in range(bpnet.num_nodes):
        current_ax = ax[i]
        current_ax.set_title(f'Weights ({mode}) Node {i}')
        num_bins = int(np.rint(2.0 / bin_size))
        scores = np.zeros((num_bins, num_bins))
        counts = np.zeros((num_bins, num_bins))

        weights = wgt_to_use[i].flatten()
        weights /= weights.sum()

        particles = msgs[i].reshape(-1, 2)
        # Convert particle coordinates to bin indices
        bin_coords = torch.clamp(((particles + 1.0) / bin_size), 0, num_bins - 1).long()

        for idx, coord in enumerate(bin_coords):
            scores[coord[1], coord[0]] += weights[idx]
            counts[coord[1], coord[0]] += 1
        
        counts[counts == 0] = 1
        out = np.divide(scores, counts)
        out = out / out.max()

        out_c = plt.cm.colors.LinearSegmentedColormap.from_list('custom', [(0,0,0,0), colors(i)])(out)
        
        current_ax.imshow(unnormed[to_show].permute(1,2,0).cpu(), extent=[-1,1,-1,1])
        current_ax.imshow(out_c, extent=[-1,1,-1,1])

    for i in range(bpnet.num_nodes, len(ax)): ax[i].set_visible(False)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=fig.dpi); plt.close('all')
    else:
        plt.show()

def plot_message_density(bpnet, grid, x, target_node, source_node, unnormed=None, to_show=0, est_bounds=1.0, fname=None):
    """
    Plots the density of a single message from a source_node to a target_node.
    This is a more general replacement for the original `plot_msg` function.

    Assumes `bpnet.neighbors` maps a target node to a list of its source neighbor nodes.
    """
    if not hasattr(bpnet, 'neighbors') or target_node not in bpnet.neighbors:
        raise AttributeError("`bpnet.neighbors` attribute not found or target_node not in it.")
    try:
        source_idx = bpnet.neighbors[target_node].index(source_node)
    except ValueError:
        raise ValueError(f"Node {source_node} is not a neighbor of node {target_node}.")

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_title(f'Message: {source_node} -> {target_node}')
    if unnormed is None: unnormed = x

    # Extract the relevant message particles and combine their weights
    msg_particles = bpnet.message_particles[target_node][to_show, source_idx, :, :]
    wgt_unary = bpnet.message_weights_unary[target_node][to_show, source_idx, :]
    wgt_neigh = bpnet.message_weights_neigh[target_node][to_show, source_idx, :]
    msg_weights = (wgt_unary * wgt_neigh).flatten()
    msg_weights /= msg_weights.sum()

    # Create grid and calculate density
    grid_res = grid.shape[0]
    grid_coords = torch.stack(torch.meshgrid(torch.linspace(-est_bounds, est_bounds, grid_res),
                                             torch.linspace(-est_bounds, est_bounds, grid_res),
                                             indexing='xy'), dim=-1)

    out = _calculate_density_from_particles(grid_coords, msg_particles, msg_weights).cpu()
    out = (out / out.max()).detach().numpy()

    # Use the target node's color
    colors = plt.cm.get_cmap('hsv', bpnet.num_nodes)
    color_map = plt.cm.colors.LinearSegmentedColormap.from_list('custom', [(0,0,0,0), colors(target_node)])

    ax.imshow(unnormed[to_show].cpu().permute(1,2,0), extent=[-est_bounds, est_bounds, -est_bounds, est_bounds], alpha=0.7)
    ax.imshow(np.flipud(out), extent=[-est_bounds, est_bounds, -est_bounds, est_bounds], cmap=color_map)

    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=300); plt.close(fig)
    else:
        plt.show()
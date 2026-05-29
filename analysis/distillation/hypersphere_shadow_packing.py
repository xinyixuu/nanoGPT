#!/usr/bin/env python3

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def energy_riesz(points, s=1.0):
    """
    Riesz energy with exponent s=1: sum_{i<j} 1 / ||points_i - points_j||.
    points: (m, n) tensor
    """
    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (m, m, n)
    sq_dists = (diff * diff).sum(dim=-1) + 1.0e-14    # shape (m, m)
    dists = torch.sqrt(sq_dists)
    inv_dists = 1.0 / dists
    diag_mask = torch.eye(dists.shape[0], dtype=torch.bool, device=points.device)
    inv_dists[diag_mask] = 0.0
    E = torch.triu(inv_dists, diagonal=1).sum()
    return E

def project_to_sphere(X):
    """
    In-place projection of each row of X onto the unit sphere.
    X: (k, n)
    """
    with torch.no_grad():
        norms = X.norm(dim=1, keepdim=True)
        X /= norms.clamp_min(1e-14)
    return X

def energy_with_shadows(X, s=1.0):
    """
    Build [X; -X], compute the Riesz energy among them.
    """
    X_shadow = -X
    Y = torch.cat([X, X_shadow], dim=0)  # shape (2k, n)
    return energy_riesz(Y, s=s)

def compute_min_angles_degrees(X_np):
    """
    Given X_np of shape (k, n) with unit vectors,
    compute the min angle (in degrees) that each vector has to any other.
    Returns array of shape (k,).
    """
    dotprods = X_np @ X_np.T
    dotprods = np.clip(dotprods, -1.0, 1.0)
    angles_radians = np.arccos(dotprods)
    angles_degrees = np.degrees(angles_radians)
    np.fill_diagonal(angles_degrees, np.nan)
    min_angles_deg = np.nanmin(angles_degrees, axis=1)
    return min_angles_deg

def make_angle_histogram_figure(
    X_np, iteration=0,
    label="Before optimization",
    dim=3,
    k=5,
    step_size=0.01,
    max_iter=1000
):
    """
    Returns (fig, min_angles_deg).
    fig is a matplotlib Figure plotting the histogram of min angles in degrees.
    min_angles_deg is the array of shape (k,).
    """
    min_angles_deg = compute_min_angles_degrees(X_np)

    fig, ax = plt.subplots()
    ax.hist(min_angles_deg, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Minimum angle to any other vector (degrees)")
    ax.set_ylabel("Count")

    title_str = (f"{label} (iter={iteration}, k={k}, dim={dim}, step={step_size}, max_iter={max_iter})")
    ax.set_title(title_str)

    fig.tight_layout()
    return fig, min_angles_deg

def main():
    parser = argparse.ArgumentParser(
        description="Pack k points on an n-dimensional sphere (with shadows) using Riesz energy, TensorBoard logs, optional Adam & LR decay."
    )
    parser.add_argument('--dim', type=int, default=3,
                        help='Dimension n of the hypersphere (points in R^n).')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of main vectors (2k total with shadows).')
    parser.add_argument('--outfile', type=str, default='points_on_sphere.npy',
                        help='Output .npy filename for main vectors after training.')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Number of gradient steps.')
    parser.add_argument('--step-size', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD','Adam'],
                        help='Optimizer choice.')
    parser.add_argument('--lr-min', type=float, default=None,
                        help='If set, use CosineAnnealingLR down to this min LR. If not set, no LR scheduler.')
    parser.add_argument('--lr-tmax', type=int, default=None,
                        help='T_max for the cosine decay (defaults to max_iter if used).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--before-plotfile', type=str, default='before_opt_angle_distribution.png',
                        help='Histogram plot file (before optimization).')
    parser.add_argument('--after-plotfile', type=str, default='after_opt_angle_distribution.png',
                        help='Histogram plot file (after optimization).')
    parser.add_argument('--logdir', type=str, default='logs',
                        help='Directory for TensorBoard logs.')
    parser.add_argument('--plot-every', type=int, default=1000,
                        help='Interval (iterations) between saving angle histogram plots. (0=never)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    # Build TensorBoard writer
    writer = SummaryWriter(log_dir=args.logdir)

    # If using LR scheduler, set default lr_tmax to max_iter
    if args.lr_min is not None and args.lr_tmax is None:
        args.lr_tmax = args.max_iter

    # Initialize X
    X = torch.randn((args.k, args.dim), device=device)
    X = project_to_sphere(X)

    # ---- BEFORE optimization: measure angles, save plot, log to TB
    X_init = X.detach().cpu().numpy()
    fig_before, min_angles_deg_init = make_angle_histogram_figure(
        X_init,
        iteration=0,
        label="Before optimization",
        dim=args.dim,
        k=args.k,
        step_size=args.step_size,
        max_iter=args.max_iter
    )
    # Save local .png
    fig_before.savefig(args.before_plotfile)
    plt.close(fig_before)
    print(f"Saved 'before' histogram to {args.before_plotfile}")

    # Also log to TensorBoard
    writer.add_figure("Angles/HistogramPlot_Before", fig_before, global_step=0)
    writer.add_histogram("Angles/MinAngles_deg_Before", min_angles_deg_init, global_step=0)
    # And log the energy before we do anything
    e_init = energy_with_shadows(X, s=1.0).item()
    writer.add_scalar("Energy", e_init, 0)
    writer.add_scalar("Angles/MinAngleMean_deg", min_angles_deg_init.mean(), 0)
    writer.add_scalar("Angles/MinAngleMin_deg", min_angles_deg_init.min(), 0)

    # Make X require grad
    X.requires_grad_(True)

    # Pick optimizer
    if args.optimizer == 'SGD':
        opt = torch.optim.SGD([X], lr=args.step_size)
    else:
        opt = torch.optim.Adam([X], lr=args.step_size)

    # Possibly add a cosine LR scheduler
    if args.lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.lr_tmax, eta_min=args.lr_min
        )
    else:
        scheduler = None

    # ---- Main training loop
    for i in range(1, args.max_iter + 1):
        opt.zero_grad()
        e = energy_with_shadows(X, s=1.0)
        e.backward()
        opt.step()
        project_to_sphere(X)

        if scheduler is not None:
            scheduler.step()

        # Log some stats to TensorBoard every 10 steps
        if i % 10 == 0:
            e_val = e.item()
            writer.add_scalar("Energy", e_val, i)
            X_np = X.detach().cpu().numpy()
            min_angles_deg = compute_min_angles_degrees(X_np)
            writer.add_scalar("Angles/MinAngleMean_deg", min_angles_deg.mean(), i)
            writer.add_scalar("Angles/MinAngleMin_deg", min_angles_deg.min(), i)
            writer.add_histogram("Angles/MinAngles_deg_raw", min_angles_deg, i)

            # learning rate
            if scheduler is not None:
                lr_current = scheduler.get_last_lr()[0]
            else:
                lr_current = opt.param_groups[0]['lr']
            writer.add_scalar("LR", lr_current, i)

        # Possibly also create a local .png histogram
        if args.plot_every > 0 and (i % args.plot_every == 0):
            X_np = X.detach().cpu().numpy()
            fig_mid, min_angles_deg_mid = make_angle_histogram_figure(
                X_np,
                iteration=i,
                label="During optimization",
                dim=args.dim,
                k=args.k,
                step_size=args.step_size,
                max_iter=args.max_iter
            )
            fig_filename = f"angles_iter_{i:06d}.png"
            fig_mid.savefig(fig_filename)
            plt.close(fig_mid)
            print(f"Saved histogram figure to {fig_filename}")

            # also add figure to TB
            writer.add_figure("Angles/HistogramPlot_During", fig_mid, global_step=i)
            # and a histogram of min angles
            writer.add_histogram("Angles/MinAngles_deg_during", min_angles_deg_mid, i)

    # ---- After finishing
    X_final = X.detach().cpu().numpy()
    np.save(args.outfile, X_final)
    print(f"Saved final (main) vectors to '{args.outfile}'. shape={X_final.shape}")

    fig_after, min_angles_deg_after = make_angle_histogram_figure(
        X_final,
        iteration=args.max_iter,
        label="After optimization",
        dim=args.dim,
        k=args.k,
        step_size=args.step_size,
        max_iter=args.max_iter
    )
    fig_after.savefig(args.after_plotfile)
    plt.close(fig_after)
    print(f"Saved 'after' histogram to {args.after_plotfile}")

    # Log final in TB
    writer.add_figure("Angles/HistogramPlot_After", fig_after, global_step=args.max_iter)
    writer.add_histogram("Angles/MinAngles_deg_After", min_angles_deg_after, global_step=args.max_iter)
    e_final = energy_with_shadows(X, s=1.0).item()
    writer.add_scalar("Energy", e_final, args.max_iter)
    writer.add_scalar("Angles/MinAngleMean_deg", min_angles_deg_after.mean(), args.max_iter)
    writer.add_scalar("Angles/MinAngleMin_deg", min_angles_deg_after.min(), args.max_iter)

    writer.close()
    print("Done! TensorBoard logs are in:", args.logdir)

if __name__ == "__main__":
    main()


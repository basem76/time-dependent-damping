"""
Grid Search Optimization for Time-Dependent Damping Parameters
Finds optimal κ, α, and simulation parameters to match theoretical predictions

Author: Basem Ajarmah
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.signal import hilbert
import pandas as pd
from itertools import product
import time
from datetime import datetime
import json
import os

# ============================================================================
# Core Functions
# ============================================================================

def oscillator_dynamics(t, y, kappa, alpha, omega0):
    """Single oscillator with time-dependent damping"""
    x, v = y
    damping = kappa * (1 + t)**(1 - alpha)
    dvdt = -damping * v - omega0**2 * x
    return [v, dvdt]

def solve_oscillator(kappa, alpha, omega0=1.0, t_span=(0, 1000), 
                     t_eval=None, x0=0.01):
    """Solve oscillator with given parameters"""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 50000)
    
    sol = solve_ivp(
        oscillator_dynamics, t_span, [x0, 0.0],
        args=(kappa, alpha, omega0),
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12
    )
    return sol.t, sol.y[0]

def extract_envelope(t, x):
    """Extract amplitude envelope using Hilbert transform"""
    analytic = hilbert(x)
    return np.abs(analytic)

def theoretical_decay(t, A0, kappa, alpha):
    """Theoretical decay from manuscript Eq. (7)"""
    if alpha >= 2.0:
        alpha = 1.999
    if alpha <= 0.01:
        alpha = 0.01
    exponent = -kappa/(2*(2-alpha)) * ((1 + t)**(2-alpha) - 1)
    exponent = np.clip(exponent, -700, 700)
    return A0 * np.exp(exponent)

def fit_decay(t, envelope, alpha_true):
    """Fit numerical decay to extract fitted alpha"""
    # Select fitting window (avoid initial transients and noise)
    mask = (t >= t[-1]*0.1) & (t <= t[-1]*0.9) & (envelope > 1e-8)
    if np.sum(mask) < 10:
        return np.nan, np.nan
    
    t_fit = t[mask]
    env_fit = envelope[mask]
    env_norm = env_fit / env_fit[0]
    
    try:
        def model(t, kappa, alpha):
            if alpha >= 2.0:
                alpha = 1.999
            exponent = -kappa/(2*(2-alpha)) * ((1 + t)**(2-alpha) - 1)
            return np.exp(exponent)
        
        popt, _ = curve_fit(
            model, t_fit, env_norm,
            p0=[0.02, alpha_true],
            bounds=[[0, 0.1], [1, 2.0]],
            maxfev=2000
        )
        
        kappa_fit, alpha_fit = popt
        return kappa_fit, alpha_fit
    except:
        return np.nan, np.nan

def compute_half_life(t, envelope):
    """Compute numerical half-life"""
    idx_half = np.where(envelope <= envelope[0]/2)[0]
    return t[idx_half[0]] if len(idx_half) > 0 else np.nan

def theoretical_half_life(kappa, alpha):
    """Theoretical half-life from manuscript"""
    if alpha >= 2.0:
        alpha = 1.999
    C = 1 + (2*(2-alpha)/kappa) * np.log(2)
    return C**(1/(2-alpha)) - 1

def compute_rmse(numerical, theoretical):
    """Compute Root Mean Square Error between numerical and theoretical"""
    # Interpolate theoretical to numerical time points
    t_num = numerical[0]
    env_num = numerical[1]
    
    # Generate theoretical at same time points
    env_theory = theoretical_decay(t_num, env_num[0], 0.02, 1.2)  # Will be overridden
    
    # Compute RMSE on log scale to capture decay better
    log_num = np.log10(np.maximum(env_num, 1e-15))
    log_theory = np.log10(np.maximum(env_theory, 1e-15))
    
    mask = ~np.isnan(log_num) & ~np.isinf(log_num)
    rmse = np.sqrt(np.mean((log_num[mask] - log_theory[mask])**2))
    return rmse

# ============================================================================
# Grid Search Function
# ============================================================================

def grid_search_parameters():
    """
    Perform grid search over parameter space to find optimal values
    that best match theoretical predictions
    """
    
    # ========================================================================
    # Parameter Grids
    # ========================================================================
    
    # Primary parameters to optimize
    kappa_grid = np.logspace(-3, 0, 10)  # 0.001 to 1.0
    alpha_grid = np.linspace(0.2, 1.8, 9)  # 0.2 to 1.8
    
    # Simulation parameters
    t_max_grid = [500, 1000, 2000, 5000]
    dt_grid = [0.01, 0.005, 0.001]
    x0_grid = [0.001, 0.01, 0.1]
    
    # Create all combinations
    param_combinations = list(product(
        kappa_grid, alpha_grid, t_max_grid, dt_grid, x0_grid
    ))
    
    print("="*80)
    print(f"GRID SEARCH OPTIMIZATION")
    print("="*80)
    print(f"Total combinations to test: {len(param_combinations)}")
    print(f"κ range: [{kappa_grid[0]:.4f}, {kappa_grid[-1]:.4f}]")
    print(f"α range: [{alpha_grid[0]:.1f}, {alpha_grid[-1]:.1f}]")
    print(f"t_max range: {t_max_grid}")
    print(f"dt range: {dt_grid}")
    print(f"x0 range: {x0_grid}")
    print("="*80)
    
    # Results storage
    results = []
    
    start_time = time.time()
    
    # Iterate through all combinations
    for idx, (kappa, alpha_true, t_max, dt, x0) in enumerate(param_combinations):
        if idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {idx}/{len(param_combinations)} "
                  f"({100*idx/len(param_combinations):.1f}%) "
                  f"Elapsed: {elapsed:.1f}s")
        
        try:
            # Set up time grid
            t_span = (0, t_max)
            t_eval = np.arange(0, t_max, dt)
            
            # Solve oscillator
            t, x = solve_oscillator(kappa, alpha_true, t_span=t_span, 
                                    t_eval=t_eval, x0=x0)
            envelope = extract_envelope(t, x)
            
            # Compute metrics
            t_half_num = compute_half_life(t, envelope)
            t_half_theory = theoretical_half_life(kappa, alpha_true)
            
            # Fit to extract alpha
            kappa_fit, alpha_fit = fit_decay(t, envelope, alpha_true)
            beta_true = 2 - alpha_true
            beta_fit = 2 - alpha_fit if not np.isnan(alpha_fit) else np.nan
            
            # Compute error metrics
            half_life_error = abs(t_half_num - t_half_theory) / t_half_theory * 100
            alpha_error = abs(alpha_fit - alpha_true) if not np.isnan(alpha_fit) else np.inf
            beta_error = abs(beta_fit - beta_true) if not np.isnan(beta_fit) else np.inf
            
            # Check if decay is captured (envelope drops sufficiently)
            decay_ratio = envelope[-1] / envelope[0] if len(envelope) > 0 else 1
            captured = decay_ratio < 0.1  # At least 90% decay
            
            # Store results
            results.append({
                # Input parameters
                'kappa': kappa,
                'alpha_true': alpha_true,
                't_max': t_max,
                'dt': dt,
                'x0': x0,
                
                # Output metrics
                't_half_num': t_half_num,
                't_half_theory': t_half_theory,
                'half_life_error': half_life_error,
                'kappa_fit': kappa_fit,
                'alpha_fit': alpha_fit,
                'beta_true': beta_true,
                'beta_fit': beta_fit,
                'alpha_error': alpha_error,
                'beta_error': beta_error,
                'decay_ratio': decay_ratio,
                'captured': captured,
                
                # Quality metrics
                'fitted_success': not np.isnan(alpha_fit),
                'overall_score': (1/half_life_error * 100) if half_life_error > 0 else 0
            })
            
        except Exception as e:
            print(f"Error at {idx}: {e}")
            continue
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Compute composite score (lower is better)
    # Weight: 50% alpha error, 30% half-life error, 20% decay captured
    df['composite_score'] = (
        0.5 * df['alpha_error'].fillna(100) / df['alpha_error'].max() +
        0.3 * df['half_life_error'] / df['half_life_error'].max() +
        0.2 * (1 - df['captured'].astype(float))
    )
    
    return df

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_results(df):
    """Analyze grid search results and find optimal parameters"""
    
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS ANALYSIS")
    print("="*80)
    
    # Filter successful fits
    df_success = df[df['fitted_success']].copy()
    
    print(f"\nTotal simulations: {len(df)}")
    print(f"Successful fits: {len(df_success)} ({100*len(df_success)/len(df):.1f}%)")
    
    # ========================================================================
    # Best overall parameters
    # ========================================================================
    
    # By alpha error
    best_alpha = df_success.loc[df_success['alpha_error'].idxmin()]
    print("\n" + "-"*50)
    print("BEST PARAMETERS BY ALPHA ERROR")
    print("-"*50)
    print(f"κ = {best_alpha['kappa']:.4f}")
    print(f"α_true = {best_alpha['alpha_true']:.2f}")
    print(f"t_max = {best_alpha['t_max']:.0f}")
    print(f"dt = {best_alpha['dt']:.4f}")
    print(f"x0 = {best_alpha['x0']:.4f}")
    print(f"Alpha error: {best_alpha['alpha_error']:.4f}")
    print(f"Beta error: {best_alpha['beta_error']:.4f}")
    print(f"Half-life error: {best_alpha['half_life_error']:.1f}%")
    
    # By half-life error
    best_half = df_success.loc[df_success['half_life_error'].idxmin()]
    print("\n" + "-"*50)
    print("BEST PARAMETERS BY HALF-LIFE ERROR")
    print("-"*50)
    print(f"κ = {best_half['kappa']:.4f}")
    print(f"α_true = {best_half['alpha_true']:.2f}")
    print(f"t_max = {best_half['t_max']:.0f}")
    print(f"dt = {best_half['dt']:.4f}")
    print(f"x0 = {best_half['x0']:.4f}")
    print(f"Half-life error: {best_half['half_life_error']:.1f}%")
    print(f"Alpha error: {best_half['alpha_error']:.4f}")
    print(f"Beta error: {best_half['beta_error']:.4f}")
    
    # By composite score
    best_composite = df_success.loc[df_success['composite_score'].idxmin()]
    print("\n" + "-"*50)
    print("BEST PARAMETERS BY COMPOSITE SCORE")
    print("-"*50)
    print(f"κ = {best_composite['kappa']:.4f}")
    print(f"α_true = {best_composite['alpha_true']:.2f}")
    print(f"t_max = {best_composite['t_max']:.0f}")
    print(f"dt = {best_composite['dt']:.4f}")
    print(f"x0 = {best_composite['x0']:.4f}")
    print(f"Composite score: {best_composite['composite_score']:.4f}")
    print(f"Alpha error: {best_composite['alpha_error']:.4f}")
    print(f"Half-life error: {best_composite['half_life_error']:.1f}%")
    
    # ========================================================================
    # Parameter sensitivity analysis
    # ========================================================================
    
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Best κ for each α
    print("\nOPTIMAL κ FOR EACH α:")
    print("-"*50)
    for alpha in sorted(df_success['alpha_true'].unique()):
        df_alpha = df_success[df_success['alpha_true'] == alpha]
        best_for_alpha = df_alpha.loc[df_alpha['alpha_error'].idxmin()]
        print(f"α = {alpha:.1f}: κ = {best_for_alpha['kappa']:.4f}, "
              f"error = {best_for_alpha['alpha_error']:.4f}")
    
    # Best t_max
    print("\nOPTIMAL SIMULATION TIME BY α:")
    print("-"*50)
    tmax_summary = df_success.groupby('alpha_true').apply(
        lambda x: x.loc[x['alpha_error'].idxmin()]
    )[['alpha_true', 't_max', 'alpha_error']]
    for _, row in tmax_summary.iterrows():
        print(f"α = {row['alpha_true']:.1f}: t_max = {row['t_max']:.0f}, "
              f"error = {row['alpha_error']:.4f}")
    
    # Best dt
    print("\nOPTIMAL TIME STEP BY α:")
    print("-"*50)
    dt_summary = df_success.groupby('alpha_true').apply(
        lambda x: x.loc[x['alpha_error'].idxmin()]
    )[['alpha_true', 'dt', 'alpha_error']]
    for _, row in dt_summary.iterrows():
        print(f"α = {row['alpha_true']:.1f}: dt = {row['dt']:.4f}, "
              f"error = {row['alpha_error']:.4f}")
    
    return {
        'best_alpha': best_alpha,
        'best_half': best_half,
        'best_composite': best_composite,
        'df': df,
        'df_success': df_success
    }

# ============================================================================
# Visualization Functions
# ============================================================================

def plot_optimization_results(results):
    """Create visualization of optimization results"""
    
    df = results['df']
    df_success = results['df_success']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Alpha error vs κ for different α
    ax = axes[0, 0]
    for alpha in sorted(df_success['alpha_true'].unique()):
        df_alpha = df_success[df_success['alpha_true'] == alpha]
        ax.semilogx(df_alpha['kappa'], df_alpha['alpha_error'], 
                   'o-', label=f'α={alpha:.1f}', alpha=0.7)
    ax.set_xlabel('κ')
    ax.set_ylabel('Alpha Error')
    ax.set_title('Alpha Error vs κ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Half-life error vs κ
    ax = axes[0, 1]
    for alpha in sorted(df_success['alpha_true'].unique()):
        df_alpha = df_success[df_success['alpha_true'] == alpha]
        ax.semilogx(df_alpha['kappa'], df_alpha['half_life_error'], 
                   'o-', label=f'α={alpha:.1f}', alpha=0.7)
    ax.set_xlabel('κ')
    ax.set_ylabel('Half-life Error (%)')
    ax.set_title('Half-life Error vs κ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Success rate vs parameters
    ax = axes[0, 2]
    success_by_kappa = df.groupby('kappa')['fitted_success'].mean() * 100
    success_by_alpha = df.groupby('alpha_true')['fitted_success'].mean() * 100
    
    x_kappa = np.arange(len(success_by_kappa))
    x_alpha = np.arange(len(success_by_alpha))
    
    ax.bar(x_kappa - 0.2, success_by_kappa.values, width=0.3, 
           label='by κ', alpha=0.7)
    ax.bar(x_alpha + 0.2, success_by_alpha.values, width=0.3, 
           label='by α', alpha=0.7)
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Fit Success Rate')
    ax.set_xticks(x_kappa)
    ax.set_xticklabels([f'{x:.3f}' for x in success_by_kappa.index], 
                       rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Optimal κ for each α
    ax = axes[1, 0]
    optimal_kappa = []
    alpha_vals = []
    for alpha in sorted(df_success['alpha_true'].unique()):
        df_alpha = df_success[df_success['alpha_true'] == alpha]
        best = df_alpha.loc[df_alpha['alpha_error'].idxmin()]
        optimal_kappa.append(best['kappa'])
        alpha_vals.append(alpha)
    
    ax.semilogy(alpha_vals, optimal_kappa, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('α')
    ax.set_ylabel('Optimal κ')
    ax.set_title('Optimal κ for Each α')
    ax.grid(True, alpha=0.3)
    
    # 5. Error distribution
    ax = axes[1, 1]
    ax.hist(df_success['alpha_error'].dropna(), bins=30, alpha=0.7, 
            edgecolor='black')
    ax.axvline(x=results['best_composite']['alpha_error'], 
               color='red', linestyle='--', 
               label=f"Best: {results['best_composite']['alpha_error']:.4f}")
    ax.set_xlabel('Alpha Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Alpha Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Composite score heatmap (simplified)
    ax = axes[1, 2]
    # Create pivot table of mean composite score
    pivot = df_success.pivot_table(
        values='composite_score', 
        index=pd.cut(df_success['kappa'], 5),
        columns=pd.cut(df_success['alpha_true'], 5),
        aggfunc='mean'
    )
    im = ax.imshow(pivot, cmap='viridis_r', aspect='auto')
    ax.set_xlabel('α')
    ax.set_ylabel('κ')
    ax.set_title('Composite Score (lower is better)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('optimization_results.pdf')
    plt.savefig('optimization_results.png')
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to run grid search optimization"""
    
    print("\n" + "="*80)
    print("PARAMETER OPTIMIZATION FOR TIME-DEPENDENT DAMPING MODEL")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run grid search
    df_results = grid_search_parameters()
    
    # Analyze results
    results = analyze_results(df_results)
    
    # Create visualizations
    plot_optimization_results(results)
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results
    results['df'].to_csv(f'grid_search_results_{timestamp}.csv', index=False)
    
    # Save best parameters
    best_params = {
        'by_alpha_error': {
            'kappa': float(results['best_alpha']['kappa']),
            'alpha': float(results['best_alpha']['alpha_true']),
            't_max': int(results['best_alpha']['t_max']),
            'dt': float(results['best_alpha']['dt']),
            'x0': float(results['best_alpha']['x0']),
            'alpha_error': float(results['best_alpha']['alpha_error']),
            'beta_error': float(results['best_alpha']['beta_error']),
            'half_life_error': float(results['best_alpha']['half_life_error'])
        },
        'by_half_life': {
            'kappa': float(results['best_half']['kappa']),
            'alpha': float(results['best_half']['alpha_true']),
            't_max': int(results['best_half']['t_max']),
            'dt': float(results['best_half']['dt']),
            'x0': float(results['best_half']['x0']),
            'alpha_error': float(results['best_half']['alpha_error']),
            'beta_error': float(results['best_half']['beta_error']),
            'half_life_error': float(results['best_half']['half_life_error'])
        },
        'by_composite': {
            'kappa': float(results['best_composite']['kappa']),
            'alpha': float(results['best_composite']['alpha_true']),
            't_max': int(results['best_composite']['t_max']),
            'dt': float(results['best_composite']['dt']),
            'x0': float(results['best_composite']['x0']),
            'alpha_error': float(results['best_composite']['alpha_error']),
            'beta_error': float(results['best_composite']['beta_error']),
            'half_life_error': float(results['best_composite']['half_life_error']),
            'composite_score': float(results['best_composite']['composite_score'])
        }
    }
    
    with open(f'best_parameters_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to:")
    print(f"  - grid_search_results_{timestamp}.csv")
    print(f"  - best_parameters_{timestamp}.json")
    print(f"  - optimization_results.pdf/png")
    
    # Print recommended parameters for each α
    print("\n" + "="*80)
    print("RECOMMENDED PARAMETERS FOR EACH α")
    print("="*80)
    
    df_success = results['df_success']
    for alpha in sorted(df_success['alpha_true'].unique()):
        df_alpha = df_success[df_success['alpha_true'] == alpha]
        best = df_alpha.loc[df_alpha['alpha_error'].idxmin()]
        print(f"\nα = {alpha:.1f}:")
        print(f"  κ = {best['kappa']:.4f}")
        print(f"  t_max = {best['t_max']:.0f}")
        print(f"  dt = {best['dt']:.4f}")
        print(f"  x0 = {best['x0']:.4f}")
        print(f"  Alpha error: {best['alpha_error']:.4f}")
        print(f"  Half-life error: {best['half_life_error']:.1f}%")
    
    return results

# ============================================================================
# Validation Function - Test best parameters
# ============================================================================

def validate_best_parameters(best_params):
    """Validate the best parameters by running a detailed simulation"""
    
    print("\n" + "="*80)
    print("VALIDATING BEST PARAMETERS")
    print("="*80)
    
    params = best_params['by_composite']
    
    print(f"\nTesting parameters:")
    print(f"  κ = {params['kappa']:.4f}")
    print(f"  α = {params['alpha']:.2f}")
    print(f"  t_max = {params['t_max']}")
    print(f"  dt = {params['dt']:.4f}")
    print(f"  x0 = {params['x0']:.4f}")
    
    # Run simulation
    t_span = (0, params['t_max'])
    t_eval = np.arange(0, params['t_max'], params['dt'])
    
    t, x = solve_oscillator(params['kappa'], params['alpha'], 
                            t_span=t_span, t_eval=t_eval, 
                            x0=params['x0'])
    envelope = extract_envelope(t, x)
    
    # Generate theoretical decay
    theory = theoretical_decay(t, envelope[0], params['kappa'], params['alpha'])
    
    # Create validation plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale
    ax = axes[0]
    ax.plot(t, envelope, 'b-', linewidth=2, label='Numerical')
    ax.plot(t, theory, 'r--', linewidth=2, label='Theoretical')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Decay Comparison - Linear Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale
    ax = axes[1]
    ax.semilogy(t, envelope, 'b-', linewidth=2, label='Numerical')
    ax.semilogy(t, theory, 'r--', linewidth=2, label='Theoretical')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Decay Comparison - Log Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation_best_params.pdf')
    plt.savefig('validation_best_params.png')
    plt.show()
    
    # Compute metrics
    t_half_num = compute_half_life(t, envelope)
    t_half_theory = theoretical_half_life(params['kappa'], params['alpha'])
    
    print(f"\nValidation Results:")
    print(f"  Numerical half-life: {t_half_num:.2f}")
    print(f"  Theoretical half-life: {t_half_theory:.2f}")
    print(f"  Error: {abs(t_half_num - t_half_theory)/t_half_theory*100:.1f}%")
    
    return fig

# ============================================================================
# Run optimization
# ============================================================================

if __name__ == "__main__":
    # Run full grid search
    results = main()
    
    # Validate best parameters
    validate_best_parameters(results['best_parameters'])
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - READY FOR MANUSCRIPT FIGURES")
    print("="*80)

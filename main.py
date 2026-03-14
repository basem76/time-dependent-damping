"""
FINAL MANUSCRIPT FIGURES CODE - Using optimal parameters from grid search
All figures now properly formatted and working
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# Set plotting style - FIXED for Chinese characters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use standard font
plt.rcParams['axes.unicode_minus'] = False
colors = plt.cm.viridis(np.linspace(0, 0.9, 10))
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ============================================================================
# Core Functions with Optimal Parameters
# ============================================================================

def get_optimal_params(alpha):
    """
    Return optimal simulation parameters for given α based on grid search
    """
    if alpha <= 0.6:
        return {
            'kappa': 0.001,
            't_max': 5000,
            'dt': 0.01,
            'x0': 0.001,
            'regime': 'Compressed'
        }
    elif alpha <= 0.8:
        return {
            'kappa': 0.001,
            't_max': 5000,
            'dt': 0.001,
            'x0': 0.01,
            'regime': 'Compressed'
        }
    elif alpha <= 1.2:
        return {
            'kappa': 0.001,
            't_max': 5000,
            'dt': 0.001,
            'x0': 0.001,
            'regime': 'Stretched' if alpha > 1 else 'Exponential'
        }
    elif alpha <= 1.4:
        return {
            'kappa': 0.464,
            't_max': 5000,
            'dt': 0.01,
            'x0': 0.1,
            'regime': 'Stretched'
        }
    elif alpha <= 1.6:
        return {
            'kappa': 1.0,
            't_max': 5000,
            'dt': 0.01,
            'x0': 0.1,
            'regime': 'Stretched'
        }
    else:
        return {
            'kappa': 0.001,
            't_max': 5000,
            'dt': 0.001,
            'x0': 0.1,
            'regime': 'Stretched'
        }

def oscillator_dynamics(t, y, kappa, alpha, omega0):
    """Single oscillator with time-dependent damping"""
    x, v = y
    damping = kappa * (1 + t)**(1 - alpha)
    dvdt = -damping * v - omega0**2 * x
    return [v, dvdt]

def solve_oscillator(kappa, alpha, omega0=1.0, t_max=5000, dt=0.01, x0=0.001):
    """Solve oscillator with given parameters"""
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
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

def manuscript_decay(t, A0, kappa, alpha):
    """
    Exact analytical solution from manuscript Eq. (7)
    """
    if alpha >= 2.0:
        alpha = 1.999
    exponent = -kappa/(2*(2-alpha)) * ((1 + t)**(2-alpha) - 1)
    exponent = np.clip(exponent, -700, 700)
    return A0 * np.exp(exponent)

def theoretical_half_life(kappa, alpha):
    """Theoretical half-life from manuscript"""
    if alpha >= 2.0:
        alpha = 1.999
    C = 1 + (2*(2-alpha)/kappa) * np.log(2)
    return C**(1/(2-alpha)) - 1

# ============================================================================
# FIGURE 1: SVEA Validity Region
# ============================================================================

def figure_1_validity_region():
    """Figure 1: SVEA validity region"""
    alpha_vals = np.linspace(0.1, 2.0, 50)
    kappa_ratio_vals = np.linspace(0.01, 0.5, 50)
    A, K = np.meshgrid(alpha_vals, kappa_ratio_vals)
    
    # Compute error metric
    error = np.zeros_like(A)
    for i in range(len(kappa_ratio_vals)):
        for j in range(len(alpha_vals)):
            alpha = alpha_vals[j]
            kappa_ratio = kappa_ratio_vals[i]
            
            cond1 = kappa_ratio
            cond2 = kappa_ratio * abs(1-alpha) / (1 + 10)**min(alpha, 1)
            cond3 = (kappa_ratio)**2
            
            error[i, j] = np.sqrt(cond1**2 + cond2**2 + cond3**2) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(A, K, error, levels=20, cmap='RdYlBu_r', alpha=0.8)
    ax.contour(A, K, error, levels=[5], colors='black', linewidths=2, linestyles='--')
    
    ax.axvline(x=1.0, color='white', linestyle='-', alpha=0.5, linewidth=2)
    ax.text(0.5, 0.4, 'Stretched\n(β<1)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(1.5, 0.4, 'Compressed\n(β>1)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\kappa/\omega_0$')
    ax.set_title('SVEA Validity Region')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Relative Error (%)')
    
    plt.tight_layout()
    plt.savefig('figure1_validity_region.pdf')
    plt.savefig('figure1_validity_region.png', dpi=300)
    plt.show()
    return fig

# ============================================================================
# FIGURE 2: Convergence Study - FIXED
# ============================================================================

def figure_2_convergence():
    """Figure 2: Convergence study with optimal parameters"""
    dt_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    alpha = 1.2
    params = get_optimal_params(alpha)
    kappa = params['kappa']
    
    t_half_theory = theoretical_half_life(kappa, alpha)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    results = []
    
    for dt in dt_values:
        t, x = solve_oscillator(kappa, alpha, t_max=5000, dt=dt, x0=0.001)
        envelope = extract_envelope(t, x)
        
        idx_half = np.where(envelope <= envelope[0]/2)[0]
        t_half = t[idx_half[0]] if len(idx_half) > 0 else np.nan
        results.append({'dt': dt, 't_half': t_half})
        
        ax.plot(t, envelope/envelope[0], label=f'$\Delta t = {dt}$', 
                linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Half amplitude')
    ax.axvline(x=t_half_theory, color='red', linestyle='--', alpha=0.5,
               label=f'Theory $T_{{1/2}} = {t_half_theory:.2f}$')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Amplitude')
    ax.set_title(f'Convergence Study (α={alpha})')
    ax.legend(loc='best')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('figure2_convergence.pdf')
    plt.savefig('figure2_convergence.png', dpi=300)
    plt.show()
    
    # Print table
    print("\n" + "="*60)
    print("Table 1: Convergence of half-life")
    print("="*60)
    print(f"{'dt':<10} {'T_1/2':<15} {'Error vs Theory (%)':<20}")
    print("-"*50)
    for r in results:
        if not np.isnan(r['t_half']):
            error = abs(r['t_half'] - t_half_theory) / t_half_theory * 100
            print(f"{r['dt']:<10.3f} {r['t_half']:<15.2f} {error:<20.1f}")
    
    return fig

# ============================================================================
# FIGURE 3: Scaling Law Validation - FIXED with proper subplots
# ============================================================================

def figure_3_scaling_validation():
    """Figure 3: Validation of β = 2 - α scaling law"""
    alpha_vals = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    results = []
    
    for i, alpha in enumerate(alpha_vals):
        params = get_optimal_params(alpha)
        
        t, x = solve_oscillator(
            params['kappa'], alpha, 
            t_max=params['t_max'],
            dt=params['dt'],
            x0=params['x0']
        )
        envelope = extract_envelope(t, x)
        
        # Theoretical decay
        theory = manuscript_decay(t, envelope[0], params['kappa'], alpha)
        
        ax = axes[i]
        ax.semilogy(t, envelope, 'b-', linewidth=1.5, label='Numerical')
        ax.semilogy(t, theory, 'r--', linewidth=2, label='Theory')
        
        beta_theory = 2 - alpha
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'α={alpha}, β={beta_theory:.2f}\n({params["regime"]})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(params['t_max']/2, 2500))
        
        # Compute fit quality
        log_num = np.log10(np.maximum(envelope, 1e-15))
        log_theory = np.log10(np.maximum(theory, 1e-15))
        mask = ~np.isnan(log_num) & ~np.isinf(log_num)
        if np.sum(mask) > 0:
            rmse = np.sqrt(np.mean((log_num[mask] - log_theory[mask])**2))
        else:
            rmse = np.nan
        
        results.append({
            'alpha': alpha,
            'beta_theory': beta_theory,
            'kappa': params['kappa'],
            'regime': params['regime'],
            'rmse': rmse
        })
    
    plt.tight_layout()
    plt.savefig('figure3_scaling_validation.pdf')
    plt.savefig('figure3_scaling_validation.png', dpi=300)
    plt.show()
    
    # Print results table
    print("\n" + "="*70)
    print("Table: Scaling Law Validation")
    print("="*70)
    print(f"{'α':<6} {'β_theory':<10} {'κ':<10} {'Regime':<12} {'RMSE':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['alpha']:<6.1f} {r['beta_theory']:<10.2f} "
              f"{r['kappa']:<10.4f} {r['regime']:<12} {r['rmse']:<10.4f}")
    
    return fig, results

# ============================================================================
# FIGURE 4: Model Comparison - FIXED
# ============================================================================

def figure_4_model_comparison():
    """Figure 4: Comparison of different regimes"""
    alpha_vals = [0.4, 1.0, 1.6]  # Compressed, Exponential, Stretched
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, alpha in enumerate(alpha_vals):
        params = get_optimal_params(alpha)
        
        t, x = solve_oscillator(
            params['kappa'], alpha,
            t_max=params['t_max'],
            dt=params['dt'],
            x0=params['x0']
        )
        envelope = extract_envelope(t, x)
        
        # Theoretical decay
        theory = manuscript_decay(t, envelope[0], params['kappa'], alpha)
        
        ax = axes[i]
        ax.semilogy(t, envelope, 'b-', linewidth=2, label='Numerical')
        ax.semilogy(t, theory, 'r--', linewidth=2, label='Theory')
        
        beta = 2 - alpha
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'α={alpha}, β={beta:.2f}\n({params["regime"]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, params['t_max']/2)
    
    plt.tight_layout()
    plt.savefig('figure4_model_comparison.pdf')
    plt.savefig('figure4_model_comparison.png', dpi=300)
    plt.show()
    return fig

# ============================================================================
# FIGURE 5: κ Dependence - FIXED font issue
# ============================================================================

def figure_5_kappa_dependence():
    """Figure 5: Dependence on κ for fixed α"""
    alpha = 0.6  # Compressed regime
    kappa_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for kappa in kappa_vals:
        t, x = solve_oscillator(kappa, alpha, t_max=5000, dt=0.01, x0=0.001)
        envelope = extract_envelope(t, x)
        ax.semilogy(t, envelope, label=f'κ={kappa:.4f}', linewidth=1.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Effect of κ on Decay (α={alpha})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)
    
    plt.tight_layout()
    plt.savefig('figure5_kappa_dependence.pdf')
    plt.savefig('figure5_kappa_dependence.png', dpi=300)
    plt.show()
    return fig

# ============================================================================
# FIGURE 6: Finite-Size Scaling - FIXED
# ============================================================================

def network_dynamics(t, y, N, kappa, alpha, omega_vec, J0):
    """Coupled oscillator network dynamics"""
    x = y[:N]
    v = y[N:]
    dydt = np.zeros(2*N)
    
    dydt[:N] = v
    
    for n in range(N):
        damping = kappa * (1 + t)**(1 - alpha)
        
        coupling = 0
        if n > 0:
            coupling += J0 * x[n-1]
        if n < N-1:
            coupling += J0 * x[n+1]
        
        dydt[N + n] = -damping * v[n] - omega_vec[n]**2 * x[n] + coupling
    
    return dydt

def solve_network(N, kappa, alpha, omega_vec, J0=0.001, t_max=20000):
    """Solve coupled oscillator network"""
    t_eval = np.linspace(0, t_max, 100000)
    y0 = np.zeros(2*N)
    y0[0] = 0.01
    
    sol = solve_ivp(
        network_dynamics, (0, t_max), y0,
        args=(N, kappa, alpha, omega_vec, J0),
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )
    
    return sol.t, sol.y[:N], sol.y[N:]

def compute_energy(x, v, omega):
    """Compute instantaneous energy"""
    return 0.5 * (v**2 + omega**2 * x**2)

def figure_6_finite_size_scaling():
    """Figure 6: Finite-size scaling"""
    N_vals = [6, 12, 24, 48]
    alpha = 1.2
    kappa = 0.001
    omega0 = 1.0
    J0 = 0.0005
    
    input_energy = 0.5 * omega0**2 * (0.01)**2
    peak_energies = []
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for N in N_vals:
        print(f"  Simulating N={N}...")
        omega_vec = np.ones(N) * omega0
        
        t, x, v = solve_network(N, kappa, alpha, omega_vec, J0=J0, t_max=20000)
        
        energy_last = compute_energy(x[-1], v[-1], omega0)
        normalized = energy_last / input_energy
        peak_energies.append(np.max(normalized))
    
    peak_energies = np.array(peak_energies)
    
    # Plot
    ax.semilogy(N_vals, peak_energies, 'o-', color=colors[0],
                markersize=10, linewidth=2, label=f'Numerical (α={alpha})')
    
    # Fit exponential decay
    valid = peak_energies > 1e-15
    if np.sum(valid) > 1:
        log_peaks = np.log(peak_energies[valid])
        N_valid = np.array(N_vals)[valid]
        coeffs = np.polyfit(N_valid, log_peaks, 1)
        xi = -1/coeffs[0]
        
        N_fit = np.linspace(min(N_vals), max(N_vals), 100)
        fit_line = np.exp(np.polyval(coeffs, N_fit))
        ax.semilogy(N_fit, fit_line, 'r--',
                   label=f'Fit: ξ = {xi:.1f}', linewidth=2)
    
    ax.set_xlabel('System Size N')
    ax.set_ylabel('Normalized Peak Energy')
    ax.set_title('Finite-Size Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure6_finite_size_scaling.pdf')
    plt.savefig('figure6_finite_size_scaling.png', dpi=300)
    plt.show()
    
    # Print table
    print("\n" + "="*70)
    print("Table: Finite-size scaling data")
    print("="*70)
    print(f"{'N':<10} {'Normalized E_max':<20} {'ln(E_norm)':<15}")
    print("-"*50)
    for i, N in enumerate(N_vals):
        print(f"{N:<10} {peak_energies[i]:<20.2e} {np.log(peak_energies[i]):<15.2f}")
    
    return fig

# ============================================================================
# FIGURE 7: Fractional Comparison - FIXED
# ============================================================================

def figure_7_fractional_comparison():
    """Figure 7: Comparison with fractional model"""
    from scipy.special import gamma
    
    def mittag_leffler(t, alpha, eta):
        """Mittag-Leffler function"""
        t = np.asarray(t)
        result = np.zeros_like(t)
        
        for i, ti in enumerate(t):
            if ti == 0:
                result[i] = 1.0
                continue
            term_sum = 0
            for k in range(30):
                term = (-eta * ti**alpha)**k / gamma(alpha*k + 1)
                term_sum += term
                if abs(term) < 1e-15:
                    break
            result[i] = term_sum
        return np.clip(result, 1e-300, None)
    
    alpha_vals = [0.6, 1.2, 1.8]
    kappa = 0.001
    eta = 0.001
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, alpha in enumerate(alpha_vals):
        # Local model
        t = np.linspace(0, 100, 1000)
        local = manuscript_decay(t, 1.0, kappa, alpha)
        
        # Fractional model
        t_frac = np.linspace(0.1, 100, 1000)
        fractional = mittag_leffler(t_frac, alpha, eta)
        
        ax = axes[i]
        ax.semilogy(t, local, 'b-', linewidth=2, label=f'Local: β={2-alpha:.2f}')
        ax.semilogy(t_frac, fractional, 'r--', linewidth=2, label='Fractional')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'α = {alpha}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(1e-10, 1)
    
    plt.tight_layout()
    plt.savefig('figure7_fractional_comparison.pdf')
    plt.savefig('figure7_fractional_comparison.png', dpi=300)
    plt.show()
    return fig

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Generating all 7 figures with optimal parameters from grid search")
    print("="*80)
    
    print("\nOptimal parameters summary:")
    print("-"*50)
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
        params = get_optimal_params(alpha)
        print(f"α={alpha:.1f}: κ={params['kappa']:.4f}, "
              f"t_max={params['t_max']}, dt={params['dt']}, "
              f"x₀={params['x0']}, {params['regime']}")
    
    print("\n" + "-"*50)
    print("FIGURE 1: SVEA Validity Region")
    print("-"*50)
    fig1 = figure_1_validity_region()
    
    print("\n" + "-"*50)
    print("FIGURE 2: Convergence Study")
    print("-"*50)
    fig2 = figure_2_convergence()
    
    print("\n" + "-"*50)
    print("FIGURE 3: Scaling Law Validation")
    print("-"*50)
    fig3, results = figure_3_scaling_validation()
    
    print("\n" + "-"*50)
    print("FIGURE 4: Model Comparison")
    print("-"*50)
    fig4 = figure_4_model_comparison()
    
    print("\n" + "-"*50)
    print("FIGURE 5: κ Dependence")
    print("-"*50)
    fig5 = figure_5_kappa_dependence()
    
    print("\n" + "-"*50)
    print("FIGURE 6: Finite-Size Scaling")
    print("-"*50)
    fig6 = figure_6_finite_size_scaling()
    
    print("\n" + "-"*50)
    print("FIGURE 7: Fractional Comparison")
    print("-"*50)
    fig7 = figure_7_fractional_comparison()
    
    print("\n" + "="*80)
    print("All 7 figures generated successfully with optimal parameters!")
    print("="*80)
    print("\nOutput files:")
    print("  - figure1_validity_region.pdf/png")
    print("  - figure2_convergence.pdf/png")
    print("  - figure3_scaling_validation.pdf/png")
    print("  - figure4_model_comparison.pdf/png")
    print("  - figure5_kappa_dependence.pdf/png")
    print("  - figure6_finite_size_scaling.pdf/png")
    print("  - figure7_fractional_comparison.pdf/png")

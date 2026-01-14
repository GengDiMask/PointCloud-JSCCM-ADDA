import numpy as np
import matplotlib.pyplot as plt
import os

def apply_inl(x, gamma=0.01):
    """DAC Integral Non-Linearity: y = x + gamma * x^3"""
    return x + gamma * np.power(x, 3)

def apply_rapp(x, p=3.0, sat=1.0):
    """Rapp Model for Power Amplifier"""
    num = x
    den = np.power(1 + np.power(np.abs(x) / sat, 2 * p), 1 / (2 * p))
    return num / den

def apply_tanh(x, alpha=1.0, beta=1.0):
    """Tanh Model for baseline saturation"""
    return alpha * np.tanh(beta * x)

def plot_models():
    # Input range: -2.0 to 2.0 (covering typical linear and saturation regions)
    x = np.linspace(-2.0, 2.0, 1000)
    
    # Configuration
    sat = 1.0
    p = 3.0
    gamma = 0.01
    
    # 1. Output Calculation (Full Chain: INL -> PA)
    # Ideal
    y_ideal = x
    
    # Rapp with INL (Implemented in project)
    x_inl = apply_inl(x, gamma)
    y_rapp_inl = apply_rapp(x_inl, p, sat)
    
    # Rapp without INL (Pure PA)
    y_rapp_pure = apply_rapp(x, p, sat)
    
    # Tanh with INL
    y_tanh_inl = apply_tanh(x_inl, alpha=1.0, beta=1.0)

    # 2. Plotting
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Main Transfer Function (AM/AM)
    plt.subplot(2, 1, 1)
    plt.title('ADDA Nonlinearity Models (Input vs Output)', fontsize=14)
    plt.plot(x, y_ideal, 'k--', label='Ideal Linear (y=x)', alpha=0.5)
    plt.plot(x, y_rapp_inl, 'r-', linewidth=2, label=f'Rapp (p={p}, sat={sat}) + INL')
    plt.plot(x, y_tanh_inl, 'b-', linewidth=2, label='Tanh (Baseline) + INL')
    
    # Mark the saturation point
    plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=-1.0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlabel('Input Amplitude (Normalized)', fontsize=12)
    plt.ylabel('Output Amplitude (Normalized)', fontsize=12)
    
    # Subplot 2: Distortion Analysis (Difference from Ideal)
    plt.subplot(2, 1, 2)
    plt.title('Nonlinearity Distortion Error (Output - Ideal)', fontsize=14)
    
    # Rapp Error
    error_rapp = y_rapp_inl - x
    plt.plot(x, error_rapp, 'r-', label='Rapp Total Error')
    
    # INL Contribution only (Zoomed view essentially)
    error_inl_only = (x + gamma * x**3) - x
    plt.plot(x, error_inl_only, 'g-.', label=f'DAC INL Only (gamma={gamma})')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlabel('Input Amplitude', fontsize=12)
    plt.ylabel('Error Magnitude', fontsize=12)
    
    # Annotations
    text = (
        f"Parameter Settings:\n"
        f"Rapp Smoothness p = {p}\n"
        f"Saturation V_sat = {sat}\n"
        f"DAC INL gamma = {gamma}\n"
    )
    plt.figtext(0.15, 0.45, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    
    save_path = 'adda_model_plot.png'
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    plot_models()

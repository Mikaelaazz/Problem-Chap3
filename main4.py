import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

def analyze_eigensystem(A, perturbations, perturbation_type='matrix'):
    """
    Menganalisis perubahan nilai eigen terhadap berbagai perturbasi
    
    Parameters:
        A: Matriks asli
        perturbations: Daftar besar perturbasi
        perturbation_type: Jenis perturbasi ('matrix', 'column', 'element')
    """
    # Hitung nilai eigen asli
    eigvals_orig, eigvecs_orig = eig(A)
    eigvals_orig = np.sort(eigvals_orig)
    
    plt.figure(figsize=(12, 6))
    
    # Analisis untuk setiap perturbasi
    for eps in perturbations:
        A_perturbed = A.copy()
        
        # Terapkan perturbasi sesuai jenis
        if perturbation_type == 'matrix':
            A_perturbed += eps * np.random.randn(*A.shape)
        elif perturbation_type == 'column':
            col = np.random.randint(A.shape[1])
            A_perturbed[:, col] += eps * np.random.randn(A.shape[0])
        elif perturbation_type == 'element':
            i, j = np.random.randint(A.shape[0]), np.random.randint(A.shape[1])
            A_perturbed[i,j] += eps
        
        # Hitung nilai eigen terperturbasi
        eigvals_pert, _ = eig(A_perturbed)
        eigvals_pert = np.sort(eigvals_pert)
        
        # Plot perubahan nilai eigen
        plt.plot(eigvals_orig, 'bo', label='Original' if eps == perturbations[0] else "")
        plt.plot(eigvals_pert, 'rx', alpha=0.5, label=f'ε={eps:.1e}' if eps == perturbations[-1] else "")
    
    plt.title(f'Perubahan Nilai Eigen - Perturbasi {perturbation_type}')
    plt.xlabel('Indeks Nilai Eigen')
    plt.ylabel('Nilai Eigen')
    plt.legend()
    plt.grid(True)
    plt.show()
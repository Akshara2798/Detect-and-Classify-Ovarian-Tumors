import numpy as np
from scipy.ndimage import gaussian_filter, laplace, sobel
import cv2

def compute_q(image, t, epsilon=1e-6):
    """
    Computes q(x, y, t) based on the gradient magnitude and Laplacian of the image.
    
    Parameters:
    image : ndarray
        Input image for which q is computed.
    t : float
        Scale parameter controlling the diffusion.
    epsilon : float
        A small constant to avoid division by zero.
    
    Returns:
    q : ndarray
        Computed q values for the input image.
    """
    # Compute the gradient magnitude using Sobel filter
    Ix = sobel(image, axis=0)
    Iy = sobel(image, axis=1)
    grad_magnitude_sq = Ix**2 + Iy**2

    # Laplacian of the image (for curvature calculations)
    laplacian_image = laplace(image)
    
    # Calculate q(x, y, t) based on the equation in the reference
    q = np.sqrt((grad_magnitude_sq / (image**2 + epsilon)) * 
                ((t**2 * laplacian_image**2) / (1 + grad_magnitude_sq / (image**2 + epsilon))))
    
    return q

def anisotropic_diffusion_coefficient(grad_x, grad_y, alpha=0.1):
    """
    Computes anisotropic diffusion coefficient based on the gradient magnitudes in different directions.
    
    Parameters:
    grad_x : ndarray
        Gradient in the x-direction.
    grad_y : ndarray
        Gradient in the y-direction.
    alpha : float
        A parameter controlling the sensitivity of diffusion to edges (smaller alpha means more edge preservation).
    
    Returns:
    c_x, c_y : ndarray
        Anisotropic diffusion coefficients for the x and y directions.
    """
    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate the anisotropic diffusion coefficient (Perona-Malik style)
    c_x = np.exp(-(grad_x**2) / (alpha * (gradient_magnitude**2 + 1e-6)))
    c_y = np.exp(-(grad_y**2) / (alpha * (gradient_magnitude**2 + 1e-6)))
    
    return c_x, c_y

def anisotropic_speckle_reducing_filter(image, num_iterations=10, alpha=0.1):
    """
    Applies anisotropic speckle reducing filtering to the input image with anisotropic diffusion.
    
    Parameters:
    image : ndarray
        Input 2D image to be filtered.
    t : float
        Scale parameter controlling the level of smoothing.
    num_iterations : int
        Number of iterations for image evolution.
    alpha : float
        Controls the sensitivity of anisotropic diffusion to edges.
    
    Returns:
    smoothed_image : ndarray
        Filtered image after applying anisotropic diffusion.
    """
    
    t = 0.00001 # Scale-space parameter
    # num_iterations = 11  # Number of iterations for evolution
    # alpha = 0.2  # Anisotropic diffusion sensitivity parameter
    # Convert the image to float32 for processing
    smoothed_image = image.astype(np.float32)
    
    # Step 1: Convolve the original image with a Gaussian kernel G(x, y, t)
    smoothed_image = gaussian_filter(smoothed_image, sigma=np.sqrt(t))
    
    # Time evolution of the image according to the PDE
    for i in range(num_iterations):
        # Step 2: Compute the gradients in x and y directions
        Ix = sobel(smoothed_image, axis=0)
        Iy = sobel(smoothed_image, axis=1)
        
        # Step 3: Compute anisotropic diffusion coefficients based on the gradients
        c_x, c_y = anisotropic_diffusion_coefficient(Ix, Iy, alpha)
        
        # Step 4: Update the image using the anisotropic diffusion equation
        divergence_x = np.gradient(c_x * Ix)[0]
        divergence_y = np.gradient(c_y * Iy)[1]
        divergence = divergence_x + divergence_y
        
        # Update the image by evolving with time step `t`
        smoothed_image += divergence * t
    
    # Clip the values to valid range [0, 255] and convert back to uint8
    smoothed_image = np.clip(smoothed_image, 0, 255).astype(np.uint8)
    
    return smoothed_image


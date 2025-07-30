import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import norm


class GaussianProcess:
    """Gaussian Process class for Bayesian Optimization"""
    
    def __init__(self, length_scale=1.0, variance=1.0, noise_var=1e-8):
        self.length_scale = length_scale
        self.variance = variance
        self.noise_var = noise_var
    
    def rbf_kernel(self, X1, X2):
        """RBF (Gaussian) kernel function"""
        distances = cdist(X1, X2, 'euclidean')
        return self.variance * np.exp(-0.5 * distances**2 / self.length_scale**2)
    
    def posterior(self, X_train, y_train, X_test, prior_mean=0.0):
        """Compute GP posterior mean and covariance with optional non-zero prior mean"""
        
        # Handle prior mean
        if callable(prior_mean):
            m_train = prior_mean(X_train).reshape(-1, 1)
            m_test = prior_mean(X_test).reshape(-1, 1)
        else:
            m_train = np.full((len(X_train), 1), prior_mean)
            m_test = np.full((len(X_test), 1), prior_mean)
        
        # Compute covariance matrices
        K_train = self.rbf_kernel(X_train, X_train) + self.noise_var * np.eye(len(X_train))
        K_test = self.rbf_kernel(X_test, X_test)
        K_cross = self.rbf_kernel(X_train, X_test)
        
        # Compute posterior mean and covariance
        K_train_inv = np.linalg.inv(K_train)
        
        # Modified posterior mean formula to account for non-zero prior mean
        mu_post = m_test + K_cross.T @ K_train_inv @ (y_train.reshape(-1, 1) - m_train)
        mu_post = mu_post.flatten()
        
        # Posterior covariance (unchanged)
        cov_post = K_test - K_cross.T @ K_train_inv @ K_cross
        
        return mu_post, cov_post
    
    def sample_posterior(self, mu_post, cov_post, n_samples=1, seed=None):
        """Sample functions from GP posterior"""
        if seed is not None:
            np.random.seed(seed)
        
        # Add small jitter for numerical stability
        cov_post += 1e-6 * np.eye(len(cov_post))
        samples = np.random.multivariate_normal(mu_post.flatten(), cov_post, n_samples)
        return samples


class AcquisitionFunctions:
    """Collection of acquisition functions for Bayesian Optimization"""
    
    @staticmethod
    def expected_improvement(mu_post, std_post, y_best, xi=0.1):
        """Expected Improvement acquisition function"""
        improvement = mu_post - y_best - xi
        Z = improvement / std_post
        ei = improvement * norm.cdf(Z) + std_post * norm.pdf(Z)
        return ei
    
    @staticmethod
    def upper_confidence_bound(mu_post, std_post, confidence=0.99):
        """Upper Confidence Bound acquisition function"""
        z_score = norm.ppf(confidence)
        ucb = mu_post + z_score * std_post
        return ucb
    
    @staticmethod
    def thompson_sampling(sample_func):
        """Thompson sampling acquisition function - just the sampled function itself"""
        return sample_func


class BOVisualizer:
    """Visualization class for Bayesian Optimization plots"""
    
    def __init__(self, X_test, X_train, y_train, colors=None, sample_labels=None):
        self.X_test = X_test
        self.X_train = X_train
        self.y_train = y_train
        self.colors = colors or ['red', 'blue', 'green']
        self.sample_labels = sample_labels or ['Sample 1', 'Sample 2', 'Sample 3']
    
    def plot_gp_samples(self, mu_post, std_post, samples, confidence_level=2.0):
        """Plot individual GP samples with confidence intervals"""
        upper_bound = mu_post + confidence_level * std_post
        lower_bound = mu_post - confidence_level * std_post
        
        for i in range(1):  # Only plot first sample as in original
            plt.figure(figsize=(10, 6))
            
            # Plot confidence region
            plt.fill_between(self.X_test.flatten(), lower_bound, upper_bound, 
                            color='lightblue', alpha=0.3, label='95% Confidence')
            
            # Plot posterior mean
            plt.plot(self.X_test.flatten(), mu_post, 'k-', linewidth=1.5, alpha=0.5, 
                    label='Posterior Mean')
            
            # Plot the sampled function
            plt.plot(self.X_test.flatten(), samples[i], color=self.colors[i], linewidth=2.5, 
                    label=f'{self.sample_labels[i]}')
            
            # Plot training points
            plt.scatter(self.X_train.flatten(), self.y_train, color='black', s=50, zorder=5, 
                       label='Training Data')
            
            # Styling
            plt.xlim(-0.5, 11)
            plt.ylim(-4, 4)
            plt.title(f'GP with {self.sample_labels[i]}', fontsize=14)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"gp_sample_{i+1}.svg", format="svg")
            plt.show()
    
    def plot_all_thompson_samples(self, samples, prior_mean):
        """Plot all Thompson sampling acquisitions together"""
        plt.figure(figsize=(10, 6))
        
        for i in range(3):
            plt.plot(self.X_test.flatten(), samples[i], color=self.colors[i], linewidth=2, 
                    label=f'{self.sample_labels[i]}')
            
            # Mark maximum for each
            max_idx = np.argmax(samples[i])
            max_x = self.X_test.flatten()[max_idx]
            max_y = samples[i][max_idx]
            plt.scatter([max_x], [max_y], color=self.colors[i], s=100, marker='*', zorder=5)
        
        # Mark existing training points
        for x_train in self.X_train.flatten():
            plt.axvline(x=x_train, color='gray', linestyle='--', alpha=0.5)
        
        # Add prior mean reference line
        if prior_mean != 0.0:
            plt.axhline(y=prior_mean, color='purple', linestyle=':', alpha=0.7, 
                       label=f'Prior Mean = {prior_mean}')
        
        plt.xlim(-0.5, 11)
        plt.ylim(-4, 4)
        plt.title(f'All Thompson Sampling Acquisitions (Prior Mean = {prior_mean})', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('Acquisition Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("all_thompson_acquisitions.svg", format="svg")
        plt.show()
    
    def plot_expected_improvement(self, ei_acquisition, y_best):
        """Plot Expected Improvement acquisition function"""
        plt.figure(figsize=(10, 6))
        
        # Plot EI acquisition function
        plt.plot(self.X_test.flatten(), ei_acquisition, color='purple', linewidth=2.5, 
                label='Expected Improvement')
        plt.fill_between(self.X_test.flatten(), ei_acquisition, 0, color='purple', alpha=0.3)
        
        # Find and mark the maximum
        max_idx = np.argmax(ei_acquisition)
        max_x = self.X_test.flatten()[max_idx]
        max_y = ei_acquisition[max_idx]
        
        plt.scatter([max_x], [max_y], color='red', s=100, marker='*', 
                   label=f'Next sample: x={max_x:.2f}', zorder=5)
        
        # Mark existing training points
        for x_train in self.X_train.flatten():
            plt.axvline(x=x_train, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlim(-0.5, 11)
        plt.ylim(0.004, 0.1)
        plt.title(f'Expected Improvement Acquisition (y_best = {y_best:.2f})', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('Expected Improvement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("expected_improvement_acquisition.svg", format="svg")
        plt.show()
    
    def plot_upper_confidence_bound(self, ucb_acquisition):
        """Plot Upper Confidence Bound acquisition function"""
        plt.figure(figsize=(10, 6))
        
        baseline = -3.5
        
        # Plot the filled UCB acquisition function
        plt.fill_between(self.X_test.flatten(), baseline, ucb_acquisition, 
                        color='lightgreen', alpha=0.7, label='UCB (π = 0.999)')
        
        # Plot the UCB curve outline
        plt.plot(self.X_test.flatten(), ucb_acquisition, color='darkgreen', linewidth=2, 
                label='αUCB (π = 0.999)')
        
        # Find and mark the maximum
        max_idx = np.argmax(ucb_acquisition)
        max_x = self.X_test.flatten()[max_idx]
        max_y = ucb_acquisition[max_idx]
        
        # Add a downward arrow at the maximum
        plt.annotate('', xy=(max_x, max_y), xytext=(max_x, max_y + 0.4),
                    arrowprops=dict(color='black', lw=2))
        
        # Mark existing training points
        for x_train in self.X_train.flatten():
            plt.axvline(x=x_train, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        
        plt.axhline(y=baseline, color='black', linewidth=1, alpha=0.3)
        
        plt.xlim(-0.5, 11)
        plt.ylim(1.5, 5)
        plt.title('Upper Confidence Bound Acquisition (π = 0.999)', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('UCB Value')
        plt.legend(loc='upper right', frameon=False)
        plt.grid(False)
        
        # Clean styling
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig("ucb_acquisition.svg", format="svg")
        plt.show()
    
    def plot_combined_subplots(self, mu_post, std_post, samples, ei_acquisition, ucb_acquisition):
        """Plot all acquisition functions in combined subplots"""
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        
        # Calculate bounds
        confidence_level = 2.0
        upper_bound = mu_post + confidence_level * std_post
        lower_bound = mu_post - confidence_level * std_post
        baseline = -3.5
        
        # Plot 1: GP Posterior
        axs[0].fill_between(self.X_test.flatten(), lower_bound, upper_bound, 
                           color='lightblue', alpha=0.3)
        axs[0].plot(self.X_test.flatten(), mu_post, 'k-', linewidth=1.5, alpha=0.5, 
                   label='Posterior Mean')
        axs[0].scatter(self.X_train.flatten(), self.y_train, color='black', s=50, zorder=5, 
                      label='Training Data')
        axs[0].plot(self.X_test.flatten(), upper_bound, color='orange', linewidth=1.5, 
                   label='Upper Confidence Bound')
        axs[0].legend()
        axs[0].set_ylim(-1, 5)
        
        # Plot 2: Thompson Sampling
        for i in range(3):
            axs[1].plot(self.X_test.flatten(), samples[i], color=self.colors[i], linewidth=2.0, 
                       label=self.sample_labels[i])
            max_idx = np.argmax(samples[i])
            axs[1].scatter(self.X_test[max_idx], samples[i][max_idx], color=self.colors[i], 
                          s=80, marker='*')
        axs[1].legend()
        
        # Plot 3: Expected Improvement
        axs[2].plot(self.X_test.flatten(), ei_acquisition, color='purple', linewidth=2.5, 
                   label='Expected Improvement')
        axs[2].fill_between(self.X_test.flatten(), ei_acquisition, 0, color='purple', alpha=0.3)
        max_idx = np.argmax(ei_acquisition)
        axs[2].scatter(self.X_test[max_idx], ei_acquisition[max_idx], color='purple', 
                      marker='*', s=100, label='Next Sample')
        axs[2].legend()
        axs[2].set_ylim(0.00, 0.1)
        
        # Plot 4: Upper Confidence Bound
        axs[3].plot(self.X_test.flatten(), ucb_acquisition, color='darkgreen', linewidth=2, 
                   label='UCB (π=0.999)')
        axs[3].fill_between(self.X_test.flatten(), baseline, ucb_acquisition, 
                           color='lightgreen', alpha=0.7)
        axs[3].set_ylim(1, 6)
        axs[3].legend()
        
        # Clean up all subplots
        for ax in axs:
            ax.grid(True, alpha=0.3)
            for x in self.X_train.flatten():
                ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("acquisition_subplots.svg", format="svg")
        plt.show()

def main():
    """Main function to run Bayesian Optimization visualization"""
    
    # Configuration
    PRIOR_MEAN = 1.0  # Change this to any value you want (0.0 for original behavior)
    
    # Define input space
    X_test = np.linspace(-0.5, 11, 200).reshape(-1, 1)
    
    # Training data points
    X_train = np.array([[2.0], [4.0], [8.0]])
    y_train = np.array([2.0, 1, 2.9])
    
    # GP parameters
    length_scale = 0.4
    variance = 0.7
    noise_var = 1e-8
    
    # Initialize Gaussian Process
    gp = GaussianProcess(length_scale=length_scale, variance=variance, noise_var=noise_var)
    
    # Compute GP posterior
    mu_post, cov_post = gp.posterior(X_train, y_train, X_test, prior_mean=PRIOR_MEAN)
    
    # Sample functions with different seeds to ensure diversity
    seeds = [9361, 2638, 14212]
    samples = []
    
    for seed in seeds:
        sample = gp.sample_posterior(mu_post, cov_post, 1, seed=seed)
        samples.append(sample[0])
    
    samples = np.array(samples)
    
    # Calculate standard deviation for acquisition functions
    var_post = np.diag(cov_post)
    std_post = np.sqrt(var_post)
    
    # Initialize visualizer
    colors = ['red', 'blue', 'green']
    sample_labels = ['Sample 1', 'Sample 2', 'Sample 3']
    visualizer = BOVisualizer(X_test, X_train, y_train, colors, sample_labels)
    
    # Generate all plots
    print("Generating GP sample plots...")
    visualizer.plot_gp_samples(mu_post, std_post, samples)
    
    print("Generating Thompson sampling plot...")
    visualizer.plot_all_thompson_samples(samples, PRIOR_MEAN)
    
    # Calculate acquisition functions
    y_best = np.max(y_train)
    ei_acquisition = AcquisitionFunctions.expected_improvement(mu_post, std_post, y_best, xi=0.01)
    ucb_acquisition = AcquisitionFunctions.upper_confidence_bound(mu_post, std_post, confidence=0.99)
    
    print("Generating Expected Improvement plot...")
    visualizer.plot_expected_improvement(ei_acquisition, y_best)
    
    print("Generating Upper Confidence Bound plot...")
    visualizer.plot_upper_confidence_bound(ucb_acquisition)
    
    print("Generating combined subplots...")
    visualizer.plot_combined_subplots(mu_post, std_post, samples, ei_acquisition, ucb_acquisition)
    
    # Print summary statistics
    print_summary_statistics(PRIOR_MEAN, y_train, y_best, mu_post, samples, ei_acquisition, ucb_acquisition, X_test)


def print_summary_statistics(prior_mean, y_train, y_best, mu_post, samples, ei_acquisition, ucb_acquisition, X_test):
    """Print summary statistics and acquisition function comparisons"""
    
    # Find next sampling points
    ei_next_x = X_test.flatten()[np.argmax(ei_acquisition)]
    ucb_next_x = X_test.flatten()[np.argmax(ucb_acquisition)]
    
    # Thompson sampling next points (maxima of each sample)
    thompson_next_points = []
    for i in range(len(samples)):
        max_idx = np.argmax(samples[i])
        thompson_next_points.append(X_test.flatten()[max_idx])
    
    print(f"\n{'='*50}")
    print(f"BAYESIAN OPTIMIZATION SUMMARY")
    print(f"{'='*50}")
    
    print(f"\nExpected Improvement - Next point: x = {ei_next_x:.3f}")
    print(f"Upper Confidence Bound (99%) - Next point: x = {ucb_next_x:.3f}")
    
    print(f"\nConfiguration:")
    print(f"Prior mean: {prior_mean}")
    print(f"Training data: {y_train}")
    print(f"Current best y value: {y_best:.3f}")
    print(f"Posterior mean range: [{mu_post.min():.3f}, {mu_post.max():.3f}]")
    print(f"Training data range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    print(f"\nAcquisition Function Comparison:")
    print(f"Thompson Sampling suggests: {[f'{x:.2f}' for x in thompson_next_points]}")
    print(f"Expected Improvement suggests: {ei_next_x:.2f}")
    print(f"UCB (99%) suggests: {ucb_next_x:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GaussianVisualizer:
    """
    Visualizes bivariate Gaussian distributions with conditional distributions.
    Shows the joint distribution, marginals, and conditional distributions.
    """
    
    def __init__(self, mu_x=0.0, mu_y=0.0, sigma_x=2.5, sigma_y=2.5, rho=0.8):
        """
        Initialize the bivariate Gaussian parameters.
        
        Args:
            mu_x, mu_y: Means of X and Y
            sigma_x, sigma_y: Standard deviations of X and Y
            rho: Correlation coefficient between X and Y
        """
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho = rho
        
        # Calculate covariance matrix
        self.cov_xy = rho * sigma_x * sigma_y
        self.covariance_matrix = np.array([[sigma_x**2, self.cov_xy],
                                          [self.cov_xy, sigma_y**2]])
        
        # Conditional distribution parameters
        self.sigma_y_given_x = sigma_y * np.sqrt(1 - rho**2)
    
    def conditional_mean_y_given_x(self, x):
        """Calculate conditional mean of Y given X=x"""
        return self.mu_y + (self.cov_xy / self.sigma_x**2) * (x - self.mu_x)
    
    def plot_gaussian(self, x_condition=3.5, x_range=(-8, 8), y_range=(-8, 8)):
        """
        Create a comprehensive plot showing joint distribution and conditionals.
        
        Args:
            x_condition: The X value to condition on
            x_range, y_range: Plot ranges for X and Y axes
        """
        # Create grids for plotting
        x_vals = np.linspace(x_range[0], x_range[1], 300)
        y_vals = np.linspace(y_range[0], y_range[1], 400)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        pos = np.dstack((X_grid, Y_grid))
        
        # Calculate joint distribution
        joint_rv = multivariate_normal([self.mu_x, self.mu_y], self.covariance_matrix)
        Z = joint_rv.pdf(pos)
        
        # Calculate conditional mean
        conditional_mean = self.conditional_mean_y_given_x(x_condition)
        
        # Create the plot
        fig = plt.figure(figsize=(12, 4))
        gs = GridSpec(1, 3, width_ratios=[1, 4, 1], wspace=0.2)
        
        # Left panel: Marginal distribution of Y
        ax_left = fig.add_subplot(gs[0])
        self._plot_marginal_y(ax_left, y_vals, y_range)
        
        # Middle panel: Joint distribution with conditioning line
        ax_middle = fig.add_subplot(gs[1], facecolor='lightyellow')
        self._plot_joint_distribution(ax_middle, X_grid, Y_grid, Z, x_condition, 
                                    conditional_mean, x_range, y_range)
        
        # Right panel: Conditional distribution of Y given X
        ax_right = fig.add_subplot(gs[2])
        self._plot_conditional_y_given_x(ax_right, y_vals, x_condition, 
                                       conditional_mean, y_range)
        
        plt.tight_layout()
        plt.savefig("mvn.svg", format="svg")
        plt.show()
        
        return fig
    
    def _plot_marginal_y(self, ax, y_vals, y_range):
        """Plot marginal distribution of Y"""
        pdf_y = norm.pdf(y_vals, loc=self.mu_y, scale=self.sigma_y)
        ax.plot(pdf_y, y_vals, color='gray')
        ax.invert_xaxis()
        ax.text(0.1, 0.9, rf'$\mu_Y={self.mu_y}$', transform=ax.transAxes, va='top')
        ax.text(0.1, 0.8, rf'$\sigma_Y={self.sigma_y}$', transform=ax.transAxes, va='top')
        ax.set_ylim(y_range)
        ax.axis('off')
    
    def _plot_joint_distribution(self, ax, X_grid, Y_grid, Z, x_condition, 
                               conditional_mean, x_range, y_range):
        """Plot joint distribution with conditioning line"""
        levels = np.linspace(Z.min(), Z.max(), 12)
        ax.contourf(X_grid, Y_grid, Z, levels=levels, cmap='RdPu', alpha=0.8)
        ax.axvline(x_condition, color='magenta', lw=2)
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='both', which='both', direction='out')
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
    
    def _plot_conditional_y_given_x(self, ax, y_vals, x_condition, conditional_mean, y_range):
        """Plot conditional distribution of Y given X"""
        pdf_y_given_x = norm.pdf(y_vals, loc=conditional_mean, scale=self.sigma_y_given_x)
        ax.plot(pdf_y_given_x, y_vals, color='magenta')
        ax.text(0.1, 0.9,
                rf'$\mu_{{Y|X={x_condition:.2f}}}={conditional_mean:.2f}$',
                transform=ax.transAxes, va='top')
        ax.text(0.1, 0.8,
                rf'$\sigma_{{Y|X}}={self.sigma_y_given_x:.2f}$',
                transform=ax.transAxes, va='top')
        ax.set_ylim(y_range)
        ax.axis('off')


class GaussianProcessVisualizer:
    """
    Visualizes Gaussian Process prior and posterior distributions.
    Shows sample functions, confidence intervals, and the effect of observations.
    """
    
    def __init__(self, length_scale=1.0, signal_variance=1.0, noise_variance=1e-10):
        """
        Initialize the Gaussian Process parameters.
        
        Args:
            length_scale: RBF kernel length scale parameter
            signal_variance: Signal variance (kernel amplitude)
            noise_variance: Observation noise variance
        """
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        
        # Create the kernel
        self.kernel = C(signal_variance) * RBF(length_scale=length_scale)
        
        # Set up matplotlib style
        rcParams['figure.figsize'] = 10, 12
        rcParams['font.size'] = 12
    
    def plot_gp_prior(self, x_range=(-5, 5), n_points=100, n_samples=15, random_state=42):
        """
        Plot the GP prior distribution with sample functions.
        
        Args:
            x_range: Range of input values
            n_points: Number of points to evaluate
            n_samples: Number of sample functions to draw
            random_state: Random seed for reproducibility
        """
        # Create input space
        X = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)
        
        # Create GP model with zero mean
        gp = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=False)
        
        # Get prior statistics
        prior_mean = np.zeros(X.shape[0])
        prior_std = np.sqrt(np.diag(self.kernel(X)))
        
        plt.figure(figsize=(10, 6))
        
        # Plot confidence intervals
        plt.fill_between(X.ravel(), 
                        prior_mean - 1.96 * prior_std, 
                        prior_mean + 1.96 * prior_std, 
                        alpha=0.2, color='b', label='95% confidence interval')
        
        # Draw and plot sample functions
        y_samples = gp.sample_y(X, n_samples=n_samples, random_state=random_state)
        for i in range(y_samples.shape[1]):
            plt.plot(X, y_samples[:, i], alpha=0.6, linewidth=1)
        
        # Plot mean function (zero for prior)
        plt.plot(X, prior_mean, 'k--', linewidth=2, label='Mean')
        
        plt.title('Prior Distribution: Zero Mean with Sample Functions', fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(-3, 3)
        plt.legend(frameon=False)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig('prior_samples.png')
        plt.show()
        
        return X, y_samples
    
    def plot_gp_posterior(self, X_obs, y_obs, x_range=(-5, 5), n_points=100, 
                         n_samples=10, random_state=123):
        """
        Plot the GP posterior distribution given observations.
        
        Args:
            X_obs: Observation input points
            y_obs: Observation output values
            x_range: Range of input values for plotting
            n_points: Number of points to evaluate
            n_samples: Number of sample functions to draw
            random_state: Random seed for reproducibility
        """
        # Create fine grid for smooth visualization
        X = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)
        
        # Create GP model
        gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.noise_variance, 
                                    normalize_y=False)
        
        # Fit the GP model with observations
        gp.fit(X_obs, y_obs)
        
        # Create plotting grid that includes observation points
        X_plot = np.vstack([X, X_obs])
        X_plot = np.sort(X_plot, axis=0)
        
        # Make predictions
        y_pred, sigma = gp.predict(X_plot, return_std=True)
        
        # Check uncertainty at observation points
        _, sigma_obs = gp.predict(X_obs, return_std=True)
        print(f"Standard deviation at observation points: {sigma_obs}")
        
        plt.figure(figsize=(10, 6))
        
        # Plot confidence intervals
        plt.fill_between(X_plot.ravel(), 
                        y_pred.ravel() - 1.96 * sigma, 
                        y_pred.ravel() + 1.96 * sigma, 
                        alpha=0.2, color='b', label='95% confidence interval')
        
        # Plot posterior mean
        plt.plot(X_plot, y_pred, 'b-', linewidth=2, label='Posterior mean')
        
        # Plot observations
        plt.scatter(X_obs, y_obs, c='k', s=50, zorder=3, label='Observations')
        
        # Plot sample functions from posterior
        rng = np.random.RandomState(random_state)
        y_samples = gp.sample_y(X_plot, n_samples=n_samples, random_state=rng)
        for i in range(y_samples.shape[1]):
            plt.plot(X_plot, y_samples[:, i], alpha=0.2, linewidth=1, color='green')
        
        # Mark observation points with vertical lines
        for x_obs in X_obs:
            plt.axvline(x=x_obs, color='lightgray', linestyle='--', alpha=0.5)
        
        plt.title('Gaussian Process Posterior Distribution', fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(-3, 3)
        plt.legend(frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return X_plot, y_pred, sigma, gp
    
    def plot_observation_zoom(self, gp, X_obs, y_obs, obs_index=0, margin=0.2):
        """
        Create a zoomed-in plot around an observation point to verify zero uncertainty.
        
        Args:
            gp: Fitted Gaussian Process model
            X_obs: Observation input points
            y_obs: Observation output values
            obs_index: Index of observation to zoom in on
            margin: Margin around the observation point
        """
        x_point = X_obs[obs_index, 0]
        
        # Create zoom grid
        X_zoom = np.linspace(x_point - margin, x_point + margin, 50).reshape(-1, 1)
        y_zoom, sigma_zoom = gp.predict(X_zoom, return_std=True)
        
        plt.figure(figsize=(8, 4))
        plt.axvline(x=x_point, color='lightgray', linestyle='--', alpha=0.7)
        
        plt.fill_between(X_zoom.ravel(), 
                        y_zoom.ravel() - 1.96 * sigma_zoom, 
                        y_zoom.ravel() + 1.96 * sigma_zoom, 
                        alpha=0.2, color='b', label='95% confidence interval')
        plt.plot(X_zoom, y_zoom, 'b-', linewidth=2, label='Mean')
        plt.scatter(X_obs[obs_index], y_obs[obs_index], c='k', s=50, zorder=3)
        
        plt.title(f'Zoom around x={x_point} observation', fontsize=12)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run both visualizations"""
    
    print("Creating Bivariate Gaussian Visualization...")
    # Create and plot bivariate Gaussian
    biv_gauss = GaussianVisualizer(mu_x=0.0, mu_y=0.0, sigma_x=2.5, 
                                           sigma_y=2.5, rho=0.8)
    biv_gauss.plot_gaussian(x_condition=3.5)
    
    print("\nCreating Gaussian Process Visualizations...")
    # Create GP visualizer
    gp_viz = GaussianProcessVisualizer(length_scale=1.0, signal_variance=1.0)
    
    # Plot GP prior
    print("Plotting GP prior...")
    X_prior, samples_prior = gp_viz.plot_gp_prior(n_samples=15)
    
    # Define observations and plot posterior
    print("Plotting GP posterior...")
    X_obs = np.array([-4.0, 0.0]).reshape(-1, 1)
    y_obs = np.array([-1.0, 0.5]).reshape(-1, 1)
    
    X_post, y_pred, sigma, fitted_gp = gp_viz.plot_gp_posterior(X_obs, y_obs)
    
    # Create zoom plot
    print("Creating zoom plot around observation...")
    gp_viz.plot_observation_zoom(fitted_gp, X_obs, y_obs, obs_index=0)
    
    print("All visualizations complete!")


if __name__ == "__main__":
    main()
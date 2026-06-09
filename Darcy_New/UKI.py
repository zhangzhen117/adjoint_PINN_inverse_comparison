from scipy.linalg import sqrtm
import numpy as np 
from tqdm import trange 

class UKI_np:
    """This class implements uki"""
    def __init__(self, para_dim, obs_dim, gamma, sigma, alpha, delta_t = 0.5) -> None:
        """
            Para:
                para_dim: the dimension of the parameter space
                obs_dim: the dimensionl of the observation spave
                gamma: the parameter for the initial covariance
                alpha: the regularization parameter
                
        """
        self.para_dim = para_dim
        self.obs_dim = obs_dim
        self.alpha = alpha 
        self.delta_t = delta_t
        self.sigma = sigma
        self.init_cov = np.eye(para_dim)*gamma**2
      
        
        
         

    def get_sigma_points(self, mean, cov):
        """This file constructs the sigma points"""
        R = sqrtm(cov)
        points = np.zeros((self.num_sigma, self.para_dim))
        if len(self.c_weights.shape) == 1:
            points[0] = mean
            temp = np.einsum('i, ij->ij', self.c_weights, R.T)
            points[1:self.para_dim + 1] = mean + temp
            points[self.para_dim + 1:] = mean - temp
        elif len(self.c_weights.shape) == 2:
            points[0] = mean 
            points[1:] = mean + np.dot(R, self.c_weights)
        return points 

    def prediction(self, mean, cov):
        """This function constructs the prediction step"""
        mean_hat = self.alpha * mean + (1 - self.alpha) * self.init_mean
        cov_hat = self.alpha**2 * cov + self.sigma_predict
        return mean_hat, cov_hat 

    def analysis(self, forward, mean, cov, obs):
        ###prediction step
        mean_hat, cov_hat = self.prediction(mean, cov)
        ###construct ensembles
        sigma_points = self.get_sigma_points(mean_hat, cov_hat)
        y_hat = forward(sigma_points).reshape(self.num_sigma, -1)
        y_hat_mean = np.einsum('j, ji->i', self.mean_weights, y_hat)
        ###construct error
        error_y = y_hat - y_hat_mean
        cov_theta_y = np.einsum('b, bi, bj->ij', self.cov_weights,
                                sigma_points - mean_hat, error_y)
        cov_y_y = np.einsum('b, bi, bj->ij', self.cov_weights, error_y, error_y) \
            + self.sigma_analysis
        
        temp = np.dot(cov_theta_y, np.linalg.inv(cov_y_y))
        mean_next = mean_hat + np.dot(obs - y_hat_mean, temp.T)
        cov_next = cov_hat - np.dot(temp, cov_theta_y.T)
        return mean_next.squeeze(), cov_next

    def sample(self, forward, init_mean, obs, N_iter,true_forward = None,
               update_freq = 0, init_cov = None,
               unscented_transform = 'modified-2n+1'):
        """
        forward: the forward operator
        init_mean: the inital mean vector for the parameters
        obs: the observation vector
        obs_cov: the noise matrix
        N_iter: the number of uki step
        update_freq: the frequency for updating the different matrix
        unscented_transform: the type of unscented transform, including modified-2n+1
        original-2n+1, modified-n+2, original-n+2
        model_error_mean: the approximate model error obtained by deeponet model 
        model_error_cov: the approximate cov obtained by deeponet model 
        """ 
        
        self.get_weights(unscented_transform)
        self.init_mean = init_mean
        self.mean = [init_mean]
        if init_cov is None:
            init_cov = self.init_cov
        self.cov = [init_cov]
        regulizer = (self.delta_t/(1 - self.delta_t) + 1 - self.alpha**2)
        self.sigma_analysis = (1/self.delta_t) * np.eye(self.obs_dim)*self.sigma**2
        self.sigma_predict = regulizer * self.init_cov
        pbar = trange(N_iter)
        for i in pbar:
            if update_freq > 0 and (i + 1) % update_freq == 0:
                self.sigma_predict = regulizer * init_cov
            init_mean, init_cov = self.analysis(forward, init_mean, init_cov, obs) 
            self.mean.append(init_mean)
            self.cov.append(init_cov)
                
        self.mean = np.vstack(self.mean)
        true_y = true_forward(self.mean)
        predict_y = forward(self.mean)
        self.predict_error = np.linalg.norm((predict_y - obs)/self.sigma_analysis[0,0], axis = 1)/2
        self.error = np.linalg.norm((true_y - obs)/self.sigma_analysis[0,0], axis = 1)/2
        self.error = self.error.squeeze()
        index = np.nanargmin(self.error)
        print(self.error)
        print("small_error: {}, index: {}".format(self.error[index], index))
        self.Mean = self.mean
        self.Cov = self.cov
        self.Error = self.error
        self.Predict_error = self.predict_error
        
        self.error = self.error[:index+1]
        self.predict_error = self.error[:index+1]
        self.mean = self.mean[:index+1]
        self.cov = self.cov[:index + 1]
        self.index = index 
                    
        return self.mean[index], self.cov[index]

    def get_weights(self, transform):
        """This generates the weights for the sigma points"""
        if transform in ['modified-2n+1', 'original-2n+1']:
            #ensemble size
            N_ens = self.para_dim * 2 + 1
            mean_weights = np.zeros(N_ens)
            cov_weights = np.zeros(N_ens)
            kappa, beta = 0.0, 2.0
            alpha = min(np.sqrt(4/(self.para_dim + kappa)), 1.0)
            lam = alpha**2*(self.para_dim + kappa) - self.para_dim
            c_weights = np.sqrt(self.para_dim + lam)*np.ones(self.para_dim)

            mean_weights[0] = lam/(self.para_dim + lam)
            mean_weights[1:] = 1/(2*(self.para_dim + lam))
            cov_weights[0] = lam/(self.para_dim + lam) + 1 - alpha**2 + beta
            cov_weights[1:] = 1/(2*(self.para_dim + lam))
            
            if transform == 'modified-2n+1':
                mean_weights[0] = 1
                mean_weights[1:] = 0
                
        elif transform in ['original-n+2', 'modified-n+2']:
            N_ens = self.para_dim + 2
            mean_weights = np.zeros(N_ens)
            cov_weights = np.zeros(N_ens)
            c_weights = np.zeros((self.para_dim, N_ens))
            alpha = self.para_dim/(4*(self.para_dim + 1))
            IM = np.zeros((self.para_dim, self.para_dim + 1))
            IM[0] = np.array([-1,1])*np.sqrt(2*alpha)
            for i in range(1, self.para_dim):
                for j in range(i):
                    IM[i,j] = 1/np.sqrt(alpha*i*(i+1))
                IM[i, i+1] = -i/np.sqrt(alpha*i*(i+1))
            c_weights = c_weights.at[:, 1:].set(IM)
            
            if transform == 'oringinal-n+2':
                mean_weights = 1/(self.para_dim + 1)
                mean_weights[0] = 0
                cov_weights = alpha 
                cov_weights[0] = 0
            else:
                mean_weights = 0
                mean_weights[0] = 1
                cov_weights = alpha 
                cov_weights[0] = 0
        self.c_weights = c_weights 
        self.mean_weights = mean_weights
        self.cov_weights = cov_weights
        self.num_sigma = N_ens


class EKI:
    """This class implements the EKI for inverse problems."""

    def __init__(self, J, param_dim, obs_dim):
        self.J = J
        self.param_dim = param_dim
        self.obs_dim = obs_dim 
    
    def analysis(self, ensemble, y_obs, forward, R):
        """Perform the analysis step of EKI."""
        N = ensemble.shape[0]
        m_mean = np.mean(ensemble, axis=0)
        perturbations = ensemble - m_mean

        G_ensemble = np.zeros((N, self.obs_dim))

        # Compute forward model outputs for the ensemble
        for i in trange(N):
            G_ensemble[i, :] = forward(ensemble[i, :])

        G_mean = np.mean(G_ensemble, axis=0)
        G_perturbations = G_ensemble - G_mean

        # Compute covariance matrices
        C_mG = (perturbations.T @ G_perturbations) / (N - 1)
        C_GG = (G_perturbations.T @ G_perturbations) / (N - 1) + R

        # Kalman gain

        ensemble = ensemble.T + C_mG @ np.linalg.solve(C_GG, (y_obs - G_ensemble).T)

        return ensemble.T
    
    def misfit(self, ensemble, y_obs, R, forward):
        """Compute the misfit for the ensemble."""
        mean = np.mean(ensemble, axis=0)
        G_mean = forward(mean)  
        residual = y_obs - G_mean
        misfit = residual.T @ np.linalg.solve(R, residual)
        return misfit
    
    def run(self, m0_ensemble, y_obs, forward, R, n_iterations):
        """Run the EKI algorithm."""
        self.ensemble = [m0_ensemble.copy()]
        ensemble = m0_ensemble.copy()
        for it in range(n_iterations):
            ensemble = self.analysis(ensemble, y_obs, forward, R)
            misfit = self.misfit(ensemble, y_obs, R, forward)
            print(f"Iteration {it+1}/{n_iterations}, Misfit: {misfit:.4f}%%")
            self.ensemble.append(ensemble.copy())
        return ensemble
    

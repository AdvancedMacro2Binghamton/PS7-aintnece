function [LLH] = model_llh(params, data, N, T)
p.rho_1 = params(1);
p.rho_2 = params(2);
p.phi_1 = params(3);
p.phi_2 = params(4);
p.beta = params(5);
p.sigma = params(6);
p.sigma_A = params(7);
p.sigma_B = params(8);

T = min(T, length(data));

%%% Model-implied transition equations:
% state transition:

%%% Empirical log-likelihoods by particle filtering
% initialize particles according to S_0
rng(0);
lr_sim=5000;
x_dist=zeros(lr_sim+3,1);
dist_shocks=p.sigma * randn(lr_sim+3,1);
for t= 3 : lr_sim+3
    x_dist(t) = p.rho_1 * x_dist(t-1) + p.rho_2 * x_dist(t-2) +...
        p.phi_1 * dist_shocks(t-1) + p.phi_2 * dist_shocks(t-2) + dist_shocks(t);
end

particle = zeros(T, N , 6);
llhs = zeros(T,1);

init_sample = randsample(lr_sim,N);
particle(1,:,1)=x_dist(init_sample+2);
particle(1,:,2)=x_dist(init_sample+1);
particle(1,:,3)=x_dist(init_sample);
particle(1,:,4)=dist_shocks(init_sample);
particle(1,:,5)=dist_shocks(init_sample+1);
particle(1,:,6)=dist_shocks(init_sample+2);

llhs(1) = log( mean( exp( ...
        log( normpdf(log(data(1,1)), particle(1,:,1), p.sigma_A) ) + ...
        log( normpdf(data(1,2), p.beta*particle(1,:,1).^2 , p.sigma_B) ) ...
        ) ) );

% predict, filter, update particles and collect the likelihood 

for t = 2:T
    particle(t,:,2) = particle(t-1,:,1);
    particle(t,:,3) = particle(t-1,:,2);
    particle(t,:,4) = particle(t-1,:,5);
    particle(t,:,5) = particle(t-1,:,6);
    particle(t,:,6) = p.sigma * randn(1,N);
    %%% Prediction:
    particle(t,:,1) = [params(1:2) params(4) params(3) 1] * reshape(particle(t,:,2:6),[5,N]);
    %%% Filtering:
    llh = log( normpdf(log(data(t,1)), particle(t,:,1), p.sigma_A) ) + ...
        log( normpdf(data(t,2), p.beta*particle(t,:,1).^2 , p.sigma_B) );
    lh = exp(llh);
    
    weights = exp( llh - log( sum(lh) ) );
    if sum(lh)==0
        weights(:) = 1 / length(weights);
    end
    % store the log(mean likelihood)
    llhs(t) = log(mean(lh));
    
    %%% Sampling:
    particle(t,:,1) = datasample(particle(t,:,1), N, 'Weights', weights);
    
end

LLH = sum(llhs);
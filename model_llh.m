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

% long run variance:
vars=(1+p.phi_1+p.phi_2)^2/(1-p.rho_1-p.rho_2)^2*p.sigma^2;

%%% Empirical log-likelihoods by particle filtering
% initialize particles according to S_0

particles = zeros(T, N);
shocks=normrnd(0,p.sigma,T,N);
llhs = zeros(T,1);

for ii=1:2
    particles(ii, :) = normrnd(0,vars,1,N);
    llhs(ii) = log( mean( exp( ...
        log( lognpdf(data(ii,1), particles(ii,:), p.sigma_A) ) + ...
        log( normpdf(data(ii,2), p.beta*particles(ii,:).^2 , p.sigma_B) ) ...
        ) ) );
end

% predict, filter, update particles and collect the likelihood 

for t = 3:T
    %%% Prediction:
    mu=[p.rho_2 p.rho_1 p.phi_2 p.phi_1]*transpose([particles(t-2:t-1) shocks(t-2:t-1)]);
    particles(t,:)=normrnd(mu,p.sigma,1,N);
    %%% Filtering:
    llh = log( lognpdf(data(t,1), particles(t,:), p.sigma_A) ) + ...
        log( normpdf(data(t,2), p.beta*particles(t,:).^2 , p.sigma_B) );
    lh = exp(llh);
    
    weights = exp( llh - log( sum(lh) ) );
    % store the log(mean likelihood)
    llhs(t) = log(mean(lh));
    
    %%% Sampling:
    particles(t,:) = datasample(particles(t,:), N, 'Weights', weights);
    
end

LLH = sum(llhs);
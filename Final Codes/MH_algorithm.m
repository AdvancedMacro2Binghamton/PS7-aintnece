%%%% Extremely simple sample application of a Metropolis hastings algorithm
%%%% (this script) and a particle filter to approximate the likelihood
%%%% function empirically ('model_llh.m').
%%%% Instructions: Specify a "true model" to generate dummy data with the
%%%% script 'generate_data.m'. Specify priors, step sizes and number of
%%%% particles below, and run this script.

tic
% likelihood simulation parameters:
N = 1000; % number of particles
T = 400; % length of time series (given by data)

% data needs to be provided:
load data 

% priors:
prior.rho_1 = @(x) unifpdf(x,-0.5,0.5);
prior.rho_2 = @(x) unifpdf(x,-0.5,0.5);
prior.phi_1 = @(x) unifpdf(x,-0.5,0.5);
prior.phi_2 = @(x) unifpdf(x,-0.5,0.5);
prior.sigma = @(x) lognpdf(x, -1/2, 1);
prior.beta= @(x) unifpdf(x,4,7);
prior.sigma_A = @(x) lognpdf(x, -1/2, 1);
prior.sigma_B = @(x) lognpdf(x, -1/2, 1);
prior.all = @(p) log(prior.rho_1(p(1))) + log(prior.rho_2(p(2))) +...
    log(prior.phi_1(p(3))) + log(prior.phi_2(p(4))) + ...
    log(prior.sigma(p(6))) + log(prior.sigma_A(p(7))) + ...
    log(prior.sigma_B(p(8))) + log(prior.beta(p(5)));

% proposals according to random walk with parameter sd's:
prop_sig.rho_1 = 0.05;
prop_sig.rho_2 = 0.05;
prop_sig.phi_1 = 0.05;
prop_sig.phi_2 = 0.05;
prop_sig.sigma = 0.05;
prop_sig.beta = 0.05;
prop_sig.sigma_A = 0.05;
prop_sig.sigma_B = 0.05;
prop_sig.all = [prop_sig.rho_1 prop_sig.rho_2 prop_sig.phi_1 ...
    prop_sig.phi_2 prop_sig.beta prop_sig.sigma prop_sig.sigma_A ...
    prop_sig.sigma_B];

% initial values for parameters
init_params = [0.5 0.5 0.1 -0.3 5 1 0.2 1];

% length of sample
M = 5000;
acc_rate = zeros(M,1);

llhs = zeros(M,1);
parameters = zeros(M,8);
parameters(1,:) = init_params;

% evaluate model with initial parameters
log_prior = prior.all(parameters(1,:));
llh = model_llh(parameters(1,:), data, N, T);
llhs(1) = log_prior + llh;

% sample:
rng(0)
proposal_chance = log(rand(M,1));
prop_step = randn(M,8);
for m = 2:M
    % proposal draw:
    prop_param = parameters(m-1,:) + prop_step(m,:) .* prop_sig.all;
    
    % evaluate prior and model with proposal parameters:
    prop_prior = prior.all(prop_param);
    if prop_prior > -Inf % theoretically admissible proposal
        prop_llh = model_llh(prop_param, data, N, T);
        llhs(m) = prop_prior + prop_llh;
        if llhs(m) - llhs(m-1) > proposal_chance(m)
            accept = 1;
        else
            accept = 0;
        end
    else % reject proposal since disallowed by prior
        accept = 0;
    end
    
    % update parameters (or not)
    if accept
        parameters(m,:) = prop_param;
        acc_rate(m) = 1;
    else
        parameters(m,:) = parameters(m-1,:);
        llhs(m) = llhs(m-1);
    end
    
    waitbar(m/M)
end
toc
avg_acc_rate=sum(acc_rate)/length(acc_rate);
str={'\rho_1','\rho_2','\phi_1','\phi_2';'\beta','\sigma','\sigma_A','\sigma_B'};
for i=1:2
    figure('Name','Posterior Distribution');
    for j=1:4
        subplot(2,2,j);
        histogram(parameters(:,4*(i-1)+j),50);
        title(str{i,j});
    end
end
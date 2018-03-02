function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
coeff = 0.5/m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


%disp(size(X));
%disp(size(y));
%disp(size(theta));
h_theta = X*theta;
J = coeff * sum((h_theta - y) .^ 2) + lambda*coeff * sum(theta(2:end,:) .^ 2);

grad_norm = (2 * coeff * (h_theta - y)' * X)';
reg_therm = [zeros(1,size(theta,2)); 2 * lambda * coeff * theta(2:end,:)];
grad = grad_norm + reg_therm;






% =========================================================================

grad = grad(:);

end

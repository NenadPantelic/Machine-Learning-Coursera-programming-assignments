function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
coeff = lambda*0.5/m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
[J_normal,grad_normal] = costFunction(theta,X,y);
reg_sum = sum(theta(2:end,:) .^ 2);
J = J_normal + coeff * reg_sum;
grad(1) = grad_normal(1);
grad(2:end) = grad_normal(2:end)+ 2*coeff * theta(2:end);



% =============================================================

end

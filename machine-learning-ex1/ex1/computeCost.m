function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%debug printings
%
%disp(size(X));
%disp(size(y));
%disp(size(theta));
h = X*theta;
%disp(size(h));
J = (0.5/m)*sum((h-y) .^ 2);



% =========================================================================

end

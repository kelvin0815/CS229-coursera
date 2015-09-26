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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J1 = y .* log(sigmoid(X * theta));
J2 = (1 - y) .* log(1 - sigmoid(X * theta));
J_unreg = sum(-(J1 + J2)) / m;
J_reg = 0;

for i = 2 : size(theta)
    J_reg = J_reg + 0.5 * lambda * theta(i)^2 / m;
end

J = J_unreg + J_reg;

grad(1) = sum((sigmoid(X * theta) - y) .* X(:, 1)) / m;

for j = 2 : size(theta)
    grad(j) = (sum((sigmoid(X * theta) - y) .* X(:, j)) / m) + lambda * theta(j) / m;
end

% =============================================================

end

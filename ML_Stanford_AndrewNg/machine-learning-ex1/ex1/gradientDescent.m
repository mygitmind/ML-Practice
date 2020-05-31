function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    temp0 = theta(1,1);
    temp1 = theta(2,1);

    theta0_derivative = 0;
    theta1_derivative = 0;


    %Partial derivate calculation for theta0
    f = @(a) ((theta(1,1) + theta(2,1)*X(a,2)) - y(a)) 
    theta0_derivative = sum(f(1:m))

    %Partial derivative calculation for theta1
    g = @(b) (((theta(1,1) + theta(2,1).*X(b,2)) - y(b)).*X(b,2))
    theta1_derivative = sum(g(1:m))

  %  for i=1:m
  %      theta0_derivative += ((theta(1,1) + theta(2,1).*X(i,2)) - y(i))
  %      theta1_derivative += ( ( (theta(1,1) + theta(2,1).*X(i,2) ) - y(i) ) ).*X(i,2)
  %  end
    
    theta0_derivative = theta0_derivative/m;
    theta1_derivative = theta1_derivative/m; 
        
    temp0 -= alpha*theta0_derivative;
    temp1 -= alpha*theta1_derivative;

    theta(1,1) = temp0
    theta(2,1) = temp1

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

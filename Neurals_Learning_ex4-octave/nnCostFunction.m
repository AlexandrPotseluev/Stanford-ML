function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 4x3

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 4x5

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 4x3
Theta2_grad = zeros(size(Theta2)); % 4x5

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Y = full(sparse(1:m,y,1)); % 16x4 (m x num_labels)

A1 = [ones(m, 1) X]; % 16x3 (m x input_layer_size + Bias column)
Z2 = A1 * Theta1'; % 16x4 (m x hidden_layer_size)
H1 = sigmoid(Z2); % 16x4 (m x hidden_layer_size)

A2 = [ones(m, 1) H1]; % 16x5 (m x hidden_layer_size + Bias: and the same time equal to n_columns of Theta2)
Z3 = A2 * Theta2';
H2 = sigmoid(Z3);

% No Bias
Theta1_NB = Theta1(:,2:end); % 4x2 (num_labels x input_layer_size)
Theta2_NB = Theta2(:,2:end); % 4x4 (num_labels x hidden_layer_size)

R = lambda/(2*m) * ( sum(sum(Theta1_NB.^2)) + sum(sum(Theta2_NB.^2)) );
J = -1/m * sum( sum( Y .* log(H2) + (1-Y) .* log(1-H2) ) ) + R;

for i = 1:m
  
  % Step 1: forward propagation
  a1 = A1(i,:); % 1x3 (1 x n_columns of Theta1)
  a2 = A2(i,:); % 1x5 (1 x n_columns of Theta2)
  a3 = H2(i,:); % 1x4 (1 x num_labels)
  
  z2 = Z2(i,:); % 1x4 (1 x hidden_layer_size)

  % Step 2: delta calculate
  d3 = a3 - Y(i,:); % 1x4 - 1x4 = 1x4 (1 x num_labels)
  d2 = (d3 * Theta2_NB) .* sigmoidGradient(z2); % 1x4 * 4x4 .* 1x4 = 1x4
  
  % Step 3: accumulate delta
  Theta2_grad = Theta2_grad + d3' * a2; % 4x5 + 4x1 * 1x5  = 4x5
  Theta1_grad = Theta1_grad + d2' * a1; % 4x3 + 4x1 * 1x3  = 4x3
   
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;


% Gradient regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1_NB;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2_NB;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

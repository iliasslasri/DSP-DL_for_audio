function [A,E] = mylpc(X,N)
% lpc  Linear Predictor Coefficients.
%     A = lpc(X,N) finds the coefficients, A=[ 1 A(2) ... A(N+1) ], of an Nth
%     order forward linear predictor.
%  
%        Xp(n) = -A(2)*X(n-1) - A(3)*X(n-2) - ... - A(N+1)*X(n-N)
%  
%     such that the sum of the squares of the errors
%  
%        err(n) = X(n) - Xp(n)
%  
%     is minimized.  X can be a vector or a matrix.  If X is a matrix
%     containing a separate signal in each column, lpc returns a model
%     estimate for each column in the rows of A.  N specifies the order of
%     the polynomial A(z) which must be a positive integer.  N must be less
%     or equal to the length of X.  If X is a matrix, N must be less or equal
%     to the length of each column of X.
%  
%     If you do not specify a value for N, lpc uses a default N =
%     length(X)-1.
%  
%     [A,E] = lpc(X,N) returns the variance (power) of the prediction error.
%  
%     lpc uses the Levinson-Durbin recursion to solve the normal equations
%     that arise from the least-squares formulation.  This computation of the
%     linear prediction coefficients is often referred to as the
%     autocorrelation method.
%  
%     % Example:
%     %   Estimate a data series using a third-order forward predictor, and 
%     %   compare to the original signal.
%     
%     %   Create signal data as the output of an autoregressive process
%     %   driven by white noise. Use the last 4096 samples of the AR process
%     %   output to avoid start-up transients:
%     randn('state',0); 
%     noise = randn(50000,1);  % Normalized white Gaussian noise
%     x = filter(1,[1 1/2 1/3 1/4],noise);
%     x = x(45904:50000);
%     a = lpc(x,3);
%     est_x = filter([0 -a(2:end)],1,x);  % Estimated signal
%     e = x - est_x;                      % Prediction error
%     [acs,lags] = xcorr(e,'coeff');      % ACS of prediction error
%     
%     %   Compare the predicted signal to the original signal
%     plot(1:97,x(4001:4097),1:97,est_x(4001:4097),'--');
%     title('Original Signal vs. lpc Estimate');
%     xlabel('Sample Number'); ylabel('Amplitude'); grid;
%     legend('Original Signal','lpc Estimate')
%  
%     %   Look at the autocorrelation of the prediction error.
%     figure; plot(lags,acs); 
%     title('Autocorrelation of the Prediction Error');
%     xlabel('Lags'); ylabel('Normalized Value'); grid;
%  
%     See also levinson, aryule, prony, stmcb.
% 
%     Reference page for lpc

if nargin < 2
    N = length(X)-1;
end
M = length(X);
Sxx = abs(fft(X,2*M-1)).^2/M;
rxx = real(ifft(Sxx));
R = toeplitz(rxx(1:N));
r = rxx(2:N+1);
A = R\r;
A = [1;-A]';
E = A*rxx(1:N+1);
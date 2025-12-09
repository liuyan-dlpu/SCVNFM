% tic
% 5*6;
% t1 = toc
% 
% tic
% (5+5i) * (6+6i);
% t2 = toc
% 
% tic
% 5^2;
% t3 = toc

N = 1e7;
tic; for k=1:N, c = 5*6; end; toc
tic; for k=1:N, c = (5+5i)*(6+6i); end; toc

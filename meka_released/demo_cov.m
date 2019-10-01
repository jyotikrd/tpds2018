clear all;
close all;
maxNumCompThreads(1);
L = dlmread('covtype_train.csv');% input data matrix A should be sparse matrix with size n by d
[~,d] = size(L);
y = L(:,d);
L =  L(:, 1:d-1);
A = sparse(L);
disp('A loaded')
%==================== parameters
k = 128; % 64
%k = 500; % target rank
gamma = power(2,3);%  3  kernel width in RBF kernel
opts.eta = 0.1; % decide the precentage of off-diagonal blocks are set to be zero(default 0.1)
opts.noc = 2; % number of clusters(default 10)
disp ('parameters set')
%==================== obtain the approximation U and S(K \approx U*S*U^T)
t = cputime;
[U,S] = meka(A,k,gamma,opts); % main function
display('Done with meka!');
fprintf('The total time cost for meka is %f secs\n',cputime -t);

%==================== measure the relative error
display('Testing meka!');
[n,d] = size(A);
fprintf('A: n= %d, k= %d \n',n,d);
rsmp = 1000; % sample several rows in K to measure kernel approximation error
rsmpind = randsample(1:n,rsmp);
tmpK = exp(-sqdist(A(rsmpind,:),A)*gamma);
Err = norm(tmpK-(U(rsmpind',:)*S)*U','fro')/norm(tmpK,'fro')
fprintf('The relative approximation error is %f\n',Err);

%==========================  Modified by Jyotikrishna Dass for TPDS from here

%S(abs(S)<1e-4)=0;
%U(abs(U)<1e-4)=0;
[nS,dS] = size(S);
fprintf('S: n= %d, k= %d \n',nS,dS);
[nU,dU] = size(U)
fprintf('U: n= %d, k= %d \n',nU,dU);

%=============================
t = cputime;
[u s v] = svd(full(S));
[nFS,dFS] = size(full(S));
fprintf('fullS: n= %d, k= %d \n',nFS,dFS);
ss = diag(s);
sss = sqrt(ss);
Sroot = u * diag(sss);
%Sroot = sqrtm(full(S));
%Sroot(abs(Sroot)<1e-4)=0;

dataX = U*Sroot;
[n2,d2] = size(dataX);
fprintf('dataX: n= %d, k= %d \n',n2,d2);
%dataX(abs(dataX)<1e-4)=0;
%spy(dataX);

dataX = [dataX y];
fprintf('The total time cost for SVD and dataX creation is %f secs\n',cputime -t);
%dlmwrite('U.csv', full(U));
%dlmwrite('S.csv', full(S));
dlmwrite('dataX_cov.csv', full(dataX));
fprintf('***************************\n');   

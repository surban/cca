if true
    %% 3d basis   
    baseA = [0, 1, 0;
             1, 0, 0;
             0, 0, 1];
    baseB = [0, 0, 1;
             0, 1, 0;
             1, 0, 0];
    latent = rand(3,500);
    n=3;
else     
    %% 1d basis 
    baseA = [0;
             1;
             2];
    baseB = [1;
             0;
             1];
    latent = rand(1,50);
    n=1;
end

%% cca
x = baseA * latent;
y = baseB * latent;

[A, B] = cca(x, y, n);

A'*x(:,1:10)
B'*y(:,1:10)

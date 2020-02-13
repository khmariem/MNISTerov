data = load('mnist_train.csv');

labels = data(:,1);
y = zeros(10,60000); %Correct outputs vector
for i = 1:60000
    y(labels(i)+1,i) = 1;
end

images = data(:,2:785);
images = images/255;

images = images'; %Input vectors

hn1 = 80; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer

%Initializing weights and biases
w12 = randn(hn1,784)*sqrt(2/784);
w23 = randn(hn2,hn1)*sqrt(2/hn1);
w34 = randn(10,hn2)*sqrt(2/hn2);
b12 = randn(hn1,1);
b23 = randn(hn2,1);
b34 = randn(10,1);

%learning rate
eta = 0.0058;

%Initializing errors and gradients
error4 = zeros(10,1);
error3 = zeros(hn2,1);
error2 = zeros(hn1,1);
errortot4 = zeros(10,1);
errortot3 = zeros(hn2,1);
errortot2 = zeros(hn1,1);
grad4 = zeros(10,1);
grad3 = zeros(hn2,1);
grad2 = zeros(hn1,1);

epochs = 50;

m = 10; %Minibatch size
t=0;

for k = 1:epochs %Outer epoch loop
    
    batches = 1;
    tic;

    
    for j = 1:60000/m
        v1=0;
        v2=0;
        v3=0;
        v4=0;
        v5=0;
        v6=0;
        mu=0.9;
        error4 = zeros(10,1);
        error3 = zeros(hn2,1);
        error2 = zeros(hn1,1);
        errortot4 = zeros(10,1);
        errortot3 = zeros(hn2,1);
        errortot2 = zeros(hn1,1);
        grad4 = zeros(10,1);
        grad3 = zeros(hn2,1);
        grad2 = zeros(hn1,1);
    for i = batches:batches+m-1 %Loop over each minibatch
        v_prev1=v1;v_prev2=v2;v_prev3=v3;v_prev4=v4;v_prev5=v5;v_prev6=v6;
    
    %Feed forward
    a1 = images(:,i);
    z2 = w12*a1 + b12;
    a2 = elu(z2);
    z3 = w23*a2 + b23;
    a3 = elu(z3);
    z4 = w34*a3 + b34;
    a4 = elu(z4); %Output vector
    
    %backpropagation
    error4 = (a4-y(:,i)).*elup(z4);
    error3 = (w34'*error4).*elup(z3);
    error2 = (w23'*error3).*elup(z2);
    
    errortot4 = errortot4 + error4;
    errortot3 = errortot3 + error3;
    errortot2 = errortot2 + error2;
    grad4 = grad4 + error4*a3';
    grad3 = grad3 + error3*a2';
    grad2 = grad2 + error2*a1';

    end
    
    %Gradient descent
%     w34 = w34 - eta/m*grad4;
%     w23 = w23 - eta/m*grad3;
%     w12 = w12 - eta/m*grad2;
%     b34 = b34 - eta/m*errortot4;
%     b23 = b23 - eta/m*errortot3;
%     b12 = b12 - eta/m*errortot2;
    v1=mu*v1 - eta/m *grad4;
    v2=mu*v2 - eta/m *grad3;
    v3=mu*v3 - eta/m *grad2;
    v4=mu*v4 - eta/m *errortot4;
    v5=mu*v5 - eta/m *errortot3;
    v6=mu*v6 - eta/m *errortot2;
    
    w34 = w34 - mu * v_prev1 + (1+mu)*v1;
    w23 = w23 - mu * v_prev2 + (1+mu)*v2;
    w12 = w12 - mu * v_prev3 + (1+mu)*v3;
    b34 = b34 - mu * v_prev4 + (1+mu)*v4;
    b23 = b23 - mu * v_prev5 + (1+mu)*v5;
    b12 = b12 - mu * v_prev6 + (1+mu)*v6;
    batches = batches + m;
    
    end
    t=t+toc;
    fprintf('Epochs:');
    disp(k) %Track number of epochs
    [images,y] = shuffle(images,y); %Shuffles order of the images for next epoch
end

disp('Training done!')
%Saves the parameters
save('wfour.mat','w34');
save('wthree.mat','w23');
save('wtwo.mat','w12');
save('bfour.mat','b34');
save('bthree.mat','b23');
save('btwo.mat','b12');
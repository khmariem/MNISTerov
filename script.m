%% Beginning


%% Train
% Vanilla Gradient Descent
% choix = 1;
% Accelrated Gradient
choix = 2;

dataset = load('mnist_train.csv');
labels = dataset(:,1);
y = zeros(10,60000); %Correct outputs vector
for i = 1:60000
    y(labels(i)+1,i) = 1;
end

images = dataset(:,2:785);
images = images/255;

images = images'; %Input vectors

% General parameters
t=0; %time of execution
epoch_size = 10;
nb_epochs = 50;
alpha = 0.0058;
% Weights
W12 = randn(80,784)*sqrt(2/784);
W23= randn(60,80)*sqrt(2/80);
W34=randn(10,60)*sqrt(2/60);
% biases
b12 = randn(80,1);
b23= randn(60,1);
b34=randn(10,1);

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

for i=1:nb_epochs
    tic;
    batches=1;
    for j=1:60000/m
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
        grad4 = zeros(10,1);
        grad3 = zeros(hn2,1);
        grad2 = zeros(hn1,1);
        for k=batches:batches+m-1
            
            v_prev1=v1;
            v_prev2=v2;
            v_prev3=v3;
            v_prev4=v4;
            v_prev5=v5;
            v_prev6=v6;
            
            %feed forward
            inp1 = images(:,k);
            out1 = W12*inp1 + b12;
            inp2 = elu(out1);
            out2 = W23*inp2 + b23;
            inp3 = elu(out2);
            out3 = W34*inp3 + b34;
            
            %backprop
            
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
      if choix==1
        w34 = w34 - eta/m*grad4;
        w23 = w23 - eta/m*grad3;
        w12 = w12 - eta/m*grad2;
        b34 = b34 - eta/m*errortot4;
        b23 = b23 - eta/m*errortot3;
        b12 = b12 - eta/m*errortot2;
      else
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
      end
      batches=batches+m;
        
    end
    t=t+toc;
    fprintf('Epochs:');
    disp(i) %Track number of epochs
    [images,y] = shuffle(images,y); %Shuffles order of the images for next epoch
end

disp('Training done!')

%% Test
test = load('mnist_test.csv');
labels = test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = test(:,2:785);
images = images/255;

images = images';

success = 0;
n = 10000;

for i = 1:n
out2 = elu(w12*images(:,i)+b12);
out3 = elu(w23*out2+b23);
out = elu(w34*out3+b34);
big = 0;
num = 0;
for k = 1:10
    if out(k) > big
        num = k-1;
        big = out(k);
    end
end

if labels(i) == num
    success = success + 1;
end
    

end

fprintf('Accuracy: ');
fprintf('%f',success/n*100);
disp(' %');  

%% elu function

function fr = elu(x)
    f = zeros(length(x),1);
    for i = 1:length(x)
    if x(i)>=0
        f(i) = x(i);
    else
        f(i) = 0.2*(exp(x(i))-1);
    end
    end
    fr = f;
end

    
%% Upload Dataset

clear;
clc;

News = readtable('News Popularity.xls');

%% Method 1 - Multi-Layer Perceptron Feedfroward Neural Network

% Feature Selection
news=News(:, {'n_tokens_title','n_tokens_content','day','n_unique_tokens', 'num_keywords', ...
  'self_reference_avg_sharess', 'global_sentiment_polarity', ...
  'rate_negative_words','rate_positive_words' ,'shares'});

news=table2array(news); % convert to array
news(any(isnan(news), 2), :)=[];%  Remove rows with'Type' as NaN

% Bin target data 
shares=news(:, end); %target data
bins=[0, 3000, 10000000]; %bin thresholds
news(:, end)=discretize(shares, bins); 

% Preprocess, normalise and set input and target features 
x=news(:, 1:end-1); % set input 
t=news(:, end); %set output data
x=normc(x)*100;  %normalise
t=t.'; % transpose
x=x.';  %transpose
t=full(ind2vec(t,2)); % Sort target data arrangement

% trian network
for hiddenlayernodes = 5:5:40 %train with different size of hidden layers

    for learningrate =  [0.001 0.01 0.1 0.5] %train with different learning rate
        
        %set training parameters
        forwardnet=feedforwardnet(hiddenlayernodes, 'trainlm'); % set training function
        forwardnet=init(forwardnet); % initialise network for each iteration
        
        % split training, test and validation sets
        forwardnet.divideFcn = 'dividerand'; 
        forwardnet.divideParam.trainRatio = 70/100;
        forwardnet.divideParam.valRatio = 15/100;
        forwardnet.divideParam.testRatio = 15/100;

        forwardnet.trainParam.lr= learningrate ; % set learning rate
        forwardnet.divideMode = 'sample'; 

        % Stopping criterions
        forwardnet.trainParam.epochs=200; % maximum epochs
        forwardnet.trainParam.time=600; %maximum elapsed training time
        forwardnet.trainParam.max_fail=5; % maximum validation failures
        forwardnet.trainParam.min_grad=1e-2; % minimum validation error 

       
        [forwardnet, tr]=train(forwardnet, x, t); % train network

        forwardnet.trainParam.show = 5;
        forwardnet.performFcn='mse';  % set mean squared error as performance parameter

        
        % estimate classification error at each iteration
        cv = cvpartition(39644, 'holdout', 0.2); % use 20% of dataset as test set
        xtest=x(:, test(cv));
        ttest=t(:, test(cv));

        y=forwardnet(xtest); % estimate class (output)

        yclass = vec2ind(y); % adjust format of output
        tclass = vec2ind(ttest); % adjust format of target

        epochs=tr.epoch(end); % extract epoch number that trainning finished 
        time=tr.time(end); % extract time taken to train
        mserror=mse(forwardnet,ttest,y); % extract mean squared error
         
        Results=[yclass;tclass];
        correctlyclassified=nnz(Results(1,:)==Results(2,:)); 
        errorclassifiedpercent=1-(correctlyclassified/length(Results)); %classification error
        
        % table of results
        table(hiddenlayernodes,:) = [hiddenlayernodes learningrate epochs time errorclassifiedpercent mserror];

        table( ~any(table,2), :)=[];
    end
end

%% MLP Results - Confusion Plot

%Plot confusion matrix
plotconfusion(ttest, y)% plot confusion plot for classification
set(gca,'xticklabel',{'Normal' 'Viral' 'overall'},'yticklabel',{'Normal' 'Viral' 'overall'})

%% MLP Results - Table
%Table of Results
header={'Hidden Layer Nodes', 'Learning Rate', 'Epochs', 'Time', 'Classification Error', 'MSE'};
Results = dataset({table,header{:}})

%% MLP Results - Graph

%Plot results of MSE at each hidden layer and learning rate
x=5:5:40; % hidden layer nodes
LR001=table(1:4:32,5); % learning rate of 0.001
LR01=table(2:4:32,5); % learning rate of 0.01
LR1=table(3:4:32,5); % learning rate of 0.1
LR5=table(4:4:32,5);  % learning rate of 0.5

figure
plot(x, LR001,'g', x, LR01,'b',x, LR1,'k', x, LR5, 'm', 'LineWidth', 2) % plot at each
hold on
xlabel('Hidden Layer Nodes');
ylabel('MSE');
lgd=legend('0.001','0.01', '0.1','0.5');
title(lgd,'Learning Rates');
hold off


%% Method 2 - SVM

% Feature Selection
newssvm=News(:, {'n_unique_tokens','global_sentiment_polarity','shares'}); 

newssvm=table2array(newssvm); % convert to array
newssvm(any(isnan(newssvm), 2), :)=[];%  Remove rows with'Type' as NaN

% data binning 
shares=newssvm(:, end);
bins=[0, 3000, 10000000];
newssvm(:, end)=discretize(shares, bins);

% normalise input data
x=newssvm(:, 1:end-1); 
x=normc(x)*100;
t=newssvm(:, end);

% Train SVM
tic % initialise time
Mdl=fitcsvm(x, t, 'KernelFunction','rbf','Standardize',true);
% 10-fold cross-validation 
crossvalmodel=crossval(Mdl);
Cverror=kfoldLoss(crossvalmodel) % classification error after 10-fold CV

toc % final time taken to train SVM

%%  SVM Results
%Plot support vectors 

sv1=x(Mdl.IsSupportVector,1); %first support vector
sv2=x(Mdl.IsSupportVector,2); %second support vector

figure
gscatter(x(:,1),x(:,2), t, 'br','xx' )
hold on
axis([0 0.2 0 1])
xlabel('Unique Tokens in Article')
ylabel('Sentiment Polarity of Article')
set(gca,'fontsize',6)
plot(sv1,sv2,'ko','MarkerSize',5)
legend('Normal','Viral','Support Vector')
hold off


clear;
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
labels(labels==0) = 10; 
inputData = images;
%% STEP 0: The Data Set is divided into train set and test set
nsample = floor(size(inputData,2)*0.8);   
DataTrain = inputData(:,1:nsample); % The top 80% datas and labels as the train set
labelsTrain = labels(1:nsample);
DataTest = inputData(:,nsample+1:end); % The last 20% as the test set
labelsTest = labels(nsample+1:end);
%% STEP 1: Preprocess the train data
[labelsTrain,true_order] = sort(labelsTrain);
DataTrain = DataTrain(:,true_order);
count = hist(labelsTrain,unique(labelsTrain));
data_class = cell(1,10);
label_class = cell(1,10);
stop = 0;
start = 1;
for k = 1:10
    stop = stop + count(k);
    data_class{k} = DataTrain(:,start:stop);
    label_class{k} = labelsTrain(start:stop);
    start = stop + 1;
end
%% STEP 2: Build the huffman tree on the basis of the frequencies of different numbers
m = count/length(labelsTrain);
[m,true_ord] = sort(m);
[h,e] = Huffman(m);

code_stastic = cell(10,1);
for i = 1:length(m)
    code = h{true_ord==i};
    code_stastic{i} = zeros(1,length(code));
    for j = 1:length(code)
        code_stastic{i}(j) = 1-str2double(code(j));
    end 
end
%% STEP 3: Training Process
tic
alpha_raw = 0.02;
theta_w = cell(1,10);
% g_iter = zeros(10,100);

for mm = 1:10  
%     data_org = data_class{mm};
%     data_com = data_class{10};
%     label = label_class{mm}; 
    l_w = length(code_stastic{mm})+1;
    theta_w{mm} = zeros(784,l_w-1);
    d_w = code_stastic{mm};
    %p_w = 1+zeros(size(data,2),1);
%     p_w_iter = zeros(1,1);
%     p_w_iter_com = zeros(1,1);
    
%     p_w_iter = zeros(1,1);
    iter = 1;
    delta_l = 1e-3;
    cost_iter = -5;
%     for iter = 1:90 
    while(delta_l >= 1e-4)
%         x_w = sum(data,2)/num_feat; 
        data = data_class{mm};
        num_feat = size(data,2);
        for ii = 1:num_feat
            index_data = randperm(size(data,2),1);
%             x_w = sum(data,2)/num_feat;
            x_w = data(:,index_data);
%             x_w_org = sum(data_org,2)/num_feat;
%             x_w_com = sum(data_com,2)/size(data_com,2);
%             e = 0;
            cost = 0;
            alpha = alpha_raw ;%+ 0.002/(1.0+iter+mm);
%             p_w = 1+zeros(num_feat,1);
%             p_w = 1;
%             p_w_com = 1;
            for nn = 2:l_w
                z = x_w'*theta_w{mm}(:,nn-1);% m*1 array
                f = sigmoid(z);% m*1 array
                g = (1-d_w(nn-1)-f)*alpha;% m*1 array
                
%                 z_org = x_w_org'*theta_w{mm}(:,nn-1);
%                 f_org = sigmoid(z_org);% m*1 array
%                 z_com = x_w_com'*theta_w{mm}(:,nn-1);
%                 f_com = sigmoid(z_com);
                %             e = e + theta_w{mm}(:,nn-1)*g';% 784*m matrix
%                 e = e + g*theta_w{mm}(:,nn-1);
                %             theta_w{mm}(:,nn-1) = theta_w{mm}(:,nn-1) + x_w*g/count(mm);
                theta_w{mm}(:,nn-1) = theta_w{mm}(:,nn-1) + g*x_w;
                cost = cost + (1-d_w(nn-1))*log(f) + d_w(nn-1)*log(1-f);
%                 p_w = p_w.*((f_org.^(1-d_w(nn-1))).*((1-f_org).^d_w(nn-1)));
%                 p_w_com = p_w_com.*((f_com.^(1-d_w(nn-1))).*((1-f_com).^d_w(nn-1)));
              
            end
            data(:,index_data) = [];
            cost_iter = [cost_iter cost];
%             p_w_iter = [p_w_iter p_w];
%             p_w_iter_com = [p_w_iter_com p_w_com];
            
            %         g_iter(mm,iter) = sum(g)/count(mm);
            
%             p_sum = sum(p_w_iter)/num_feat;
%             p_sum_com = sum(p_w_iter_com)/size(data_com,2);
%             delta_p = p_sum(iter)-p_sum(iter-1);
            %         data = data + e;
%             delta_p = p_w_iter(iter)-p_w_iter(iter-1);
            
%             data = data + repmat(e,1,num_feat); % 784*m
%             data(:,index_data) = data(:,index_data) + e;
%             if delta_p <= 0
%                 delta_p = 1e-3;
%             end
        end
        iter = iter +1;
        delta_l = cost_iter(end) - cost_iter(end-1);
%     end
    end
    
end
toc
%% STEP 4: Test the code
x_w = DataTest;
p_w = 1+zeros(size(DataTest,2),10);
% data_test = zeros(784,10);
% for jj = 1:10
%     data_test(:,jj) = sum(data_class{jj},2)/size(data_class{jj},2);
% end
for mm = 1:10
    l_w = length(code_stastic{mm})+1;
    d_w = code_stastic{mm};
    for nn = 2:l_w
        z = x_w'*theta_w{mm}(:,nn-1);% m*1 array
        f = sigmoid(z);% m*1 array
        p_w(:,mm) = p_w(:,mm).*((f.^(1-d_w(nn-1))).*((1-f).^d_w(nn-1)));
    end
    
    
end
[max_p,index]=max(p_w,[],2);
index = index';
% label = (1:10)';
acc = mean(labelsTest(:) == index(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);


function [S_weight,S_number] = closest_neighbors(Z,num_good,num_ordinary)

% num_good: 8 here, the number of good neighbors
% num_ordinary: 20 here, the number of ordinary neighbors
% the optimal parameter set varies with different datasets
% Z: 相似性矩阵
% num_good: 好邻居的数量
% num_ordinary: 普通邻居的数量
% 该函数的作用是选择最近邻系数，即从相似性矩阵Z中选择最相关的邻居样本

Z = Z - diag(diag(Z)); % 去除相似性矩阵Z的对角线元素

z=sum(Z');
% 计算相似性矩阵 Z 的每一行元素的和，并将结果存储在变量 z 中。
w = zeros(size(Z,1));
% disp(size(Z,1));
for i=1:size(Z,1)
    w(i,:)=Z(i,:)./z(i); % 计算每个样本的邻居权重w
end

S_number_temp = zeros(size(w,1),num_ordinary);
for i=1:size(w,1)
    b=w(i,:)+w(:,i)';% 计算每个样本的C矩阵，即每个样本与其他样本的相似性加权和
    % 使用 sort 函数对向量进行排序，并返回排序后的值和相应的索引
    [~, sorted_indices] = sort(b, 'descend');
    % 取出排序后的前 n 个索引
    S_number_temp(i,:) = sorted_indices(1:num_ordinary);
end

num=0;
S_weight = zeros(size(w,1));
S_number = zeros(size(w,1),num_good);
%disp(size(w,1))

for i=1:size(w,1)
    b=w(i,:)+w(:,i)';% 重新初始化C矩阵

    for j=1:num_good 
        for k=1:size(w,1)
            [p,q]=max(b); % 继续寻找下一个最大值
            b(q)=0; % 将已选过的邻居对应的C矩阵位置置零，避免重复选择
            if  find(S_number_temp(q,:)==i)% 判断该邻居是否已经在普通邻居中出现过
                S_weight(i,q)=p; % 将该邻居权重记录为好邻居的权重
                S_number(i,j)=q; % 将该邻居编号记录为好邻居的编号
                break;
            end  
        end
    end
end

for k=1:size(w,1)
   idx=find(S_number(k,:)==0); % 检查是否存在未选择的邻居
   S_number(k,idx)=k;% 将未选择的邻居设为自身样本编号
   S_weight(k,idx)=1;% 将未选择的邻居权重置为1
end



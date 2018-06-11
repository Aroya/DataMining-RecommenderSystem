testUser=[1];
% testUser=[1:5];
% testUser=[1:30];

data=load('datasets/small/train_small_2.txt');
% data=load('datasets\movielens\ml-100k\ml-100k\u.data');
users=data(:,1);
goods=data(:,2);
scores=data(:,3);

%good matrix
GoodSparse=sparse(users,[1:length(goods)],goods);

%score matrix
Sparse=sparse(users,goods,scores);

counter=0;
err_base=[];

normSet=[];
for i=Sparse
    normSet=[normSet,norm(i,2)];
end

for thisUser=testUser
    Goods=GoodSparse(thisUser,:);
    counter=counter+nnz(Goods);
    for thisGood=Goods
        if thisGood==0
            continue
        end
        %predict thisGood
        this=Sparse(:,thisGood);
        this(thisUser)=0;
        thisNorm=norm(this,2);
        similiarities=0;
        predictValue=0;

        %find most similiar good
        for search=Goods
            if search==thisGood || search==0
                continue
            end
            thisSearch=Sparse(:,search);
            similiarity=dot(thisSearch,this)/(thisNorm*normSet(search));
            similiarities=similiarities+similiarity;
            predictValue=predictValue+similiarity*Sparse(thisUser,search);
        end
        %similiar caculated, find the maximum
        value=predictValue/similiarities;
        thisExact=Sparse(thisUser,thisGood);
        thisErr=abs(value-thisExact);
%         log_msg=sprintf('Predict:%d\tExact:%d\tErr:%d\n',value,full(thisExact),thisErr);
%         disp(log_msg);
        err_base=[err_base,thisErr];
    end
end
save('IBCF-err.csv','err_base','-ascii');
% err_sum=sum(err_base/length(err_base));
err_sum=mean2(err_base);
log_msg=sprintf('Average err:%d',err_sum);
disp(log_msg);
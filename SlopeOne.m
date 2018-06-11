testUsers=[1];
testUsers=[1:5];
testUsers=[1:30];

data=load('datasets/small/train_small_2.txt');
% data=load('datasets\movielens\ml-100k\ml-100k\u.data');

users=data(:,1);
goods=data(:,2);
scores=data(:,3);
generator=[1:length(goods)];

Sparse=sparse(users,goods,scores);
UserSparse=sparse(goods,generator,users);
GoodSparse=sparse(users,generator,goods);

err_base=[];
predictedItem=0;
%store the dev between i&j
dev=sparse(length(goods),length(goods),0);

for thisUser=testUsers
    thisUserGoods=GoodSparse(thisUser,:);

    %caculate all dev first
    for thisGood=thisUserGoods
        if thisGood==0
            continue
        end
        goodUsers=UserSparse(thisGood,:);
        for searchGood=thisUserGoods
            if searchGood==thisGood || searchGood==0 || dev(thisGood,searchGood)~=0
                continue
            end
            counter_user=0;
            thisDev=0;
            for goodUser=goodUsers
                if goodUser==0 || Sparse(goodUser,searchGood)==0
                    continue
                end
                counter_user=counter_user+1;
                thisDev=thisDev+Sparse(goodUser,thisGood)-Sparse(goodUser,searchGood);
            end
            dev(thisGood,searchGood)=thisDev/counter_user;
            dev(searchGood,thisGood)=-dev(thisGood,searchGood);
        end
    end

    %try to predict
    for thisGood=thisUserGoods
        if thisGood==0
            continue
        end
        goodCounter=0;
        predictValue=0;
        for otherGood=thisUserGoods
            if otherGood==thisGood || otherGood==0
                continue
            end
            goodCounter=goodCounter+1;
            predictValue=predictValue+Sparse(thisUser,otherGood)+dev(thisGood,otherGood);
        end
        predictValue=predictValue/goodCounter;
        thisErr=abs(Sparse(thisUser,thisGood)-predictValue);
        % log_msg=sprintf('Predict:%d\tExact:%d\tErr:%d\t',predictValue,full(Sparse(thisUser,thisGood)),thisErr);
        % disp(log_msg);
        err_base=[err_base,thisErr];
        predictedItem=predictedItem+1;
    end
end
testCounter=length(err_base);
log_msg=sprintf('Average error:%d\twith %d tests',sum(err_base/testCounter),testCounter);
disp(log_msg);
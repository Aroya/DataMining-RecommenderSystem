%set k
k=10;
learningRate=0.01;

data=load('datasets/small/train_small_2.txt');
% data=load('datasets\movielens\ml-100k\ml-100k\u.data');

[data_rows data_cols]=size(data);
Pnext=rand(data_rows,k);
Qnext=rand(k,data_cols);
R=data;
P=[];
Q=[];
learningRate=learningRate*2;
err_base=[];

R_Predict=Pnext*Qnext;
err=R-R_Predict;
lossBefore=realmax;
lossNow=norm(err,2);
err_base=[err_base;lossNow];
steps=1;
stepFilter=10;
while lossNow<lossBefore && steps<=20000
    lossBefore=lossNow;
    P=Pnext;
    Q=Qnext;
    Pupdate=zeros(data_rows,k);
    Qupdate=zeros(k,data_cols);
    for i=[1:data_rows]
        %derivate to P
        for j=[1:data_cols]
            for p=[1:k]
                Pupdate(i,p)=Pupdate(i,p)+err(i,j)*Q(p,j);
                Qupdate(p,j)=Qupdate(p,j)+err(i,j)*P(i,p);
            end
        end
    end
    Pupdate=Pupdate/data_cols;
    Qupdate=Qupdate/data_rows;
    Pnext=P+learningRate*Pupdate;
    Qnext=Q+learningRate*Qupdate;
    R_Predict=Pnext*Qnext;
    err=R-R_Predict;
    lossNow=norm(err,2);
    err_base=[err_base;lossNow];
    steps=steps+1;
    if mod(steps,stepFilter)==0
        log_msg=sprintf('NowLoss:%d\tIterating Times:%d',lossNow,steps);
        disp(log_msg);
    end
    if steps/stepFilter>=10
        stepFilter=stepFilter*10;
    end
end
save('temp.csv','err_base','-ascii');
log_msg=sprintf('NowLoss:%d\tIterating Times:%d',lossNow,length(err_base));
disp(log_msg);
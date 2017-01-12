########################################

function what=ridge(Xt,Yt,lambda)
    d=size(Xt)(2);
    I=eye(d);
    what=(Xt'*Xt+lambda*I)\Xt'*Yt;
end

function [lambda,what]=ridgestar(Xt,Yt,Xv,Yv,gamma=1)
    d=size(Xt)(2);
    I=eye(d);
    XtTYt=Xt'*Yt;
    XtTXt=Xt'*Xt;
    XvTXv=Xv'*Xv;
    
    f = @(lambda) norm(Xv/(XtTXt+lambda*I)*XtTYt - Yv)**2+gamma*lambda**2;
    [x,fval,info,output]=fminbnd(@(x) f(exp(x)),-20,20);
    lambda=exp(x);
    if info!=1
        printf("Error in optimization.\n")
    else
        #output
    end
    what=(XtTXt+lambda*I)\XtTYt;
end

function [lambda,what]=ridgestarfast(Xt,Yt,Xv,Yv,gamma=1)
    d=size(Xt)(2);
    I=eye(d);
    XtTYt=Xt'*Yt;
    XtTXt=Xt'*Xt;
    XvTXv=Xv'*Xv;

    k=d;
    [P,D]=eigs(XtTXt,k);
    Dvec=diag(D);
    XvP=Xv*P;
    invPXtTYt=P\XtTYt;

    #f = @(lambda) norm(XvP*diag((diag(D+lambda*I)).^(-1))*invPXtTYt - Yv)**2+gamma*lambda**2;
    f = @(lambda) norm(XvP*diag((Dvec.+lambda).^(-1))*invPXtTYt - Yv)**2+gamma*lambda**2;
    [x,fval,info,output]=fminbnd(@(x) f(exp(x)),-20,20);
    lambda=exp(x);
    if info!=1
        printf("Error in optimization.\n")
    else
        output
    end
    opts=optimset('MaxIter',100);
    [x,fval,info,output]=fminunc(@(x) f(exp(x)),0,opts);
    lambda=exp(x);
    if info!=1 && info!=2 && info!=3
        printf("Error %d in optimization.\n",info)
        output
    else
        output
    end
    what=(XtTXt+lambda*I)\XtTYt;
    #what=XvP*diag((Dvec.+lambda).^(-1))*invPXtTYt;
end

########################################

d=300;
t=300;
v=10;
epsilon=1e0;
model=@(x) x;

mu=zeros([1,d]);
sigma=1e0*eye(d);
Xt=mvnrnd(repmat(mu,t,1),sigma);
Xv=mvnrnd(repmat(mu,v,1),sigma);
wstar=mvnrnd(mu,1e1*sigma)';
Yt=arrayfun(model, Xt*wstar)+mvnrnd(zeros(1,t),epsilon*eye(t))';
Yv=arrayfun(model, Xv*wstar)+mvnrnd(zeros(1,v),epsilon*eye(v))';

I=eye(d);
XtTYt=Xt'*Yt;
XtTXt=Xt'*Xt;
XvTXv=Xv'*Xv;

########################################

#tic
#[lhat,whatlhat]=ridgestar(Xt,Yt,Xv,Yv);
#err=norm(wstar-whatlhat)
#toc
tic
[lhat,whatlhat]=ridgestarfast(Xt,Yt,Xv,Yv);
err=norm(wstar-whatlhat)
toc

lambdas=10.^[-10:0.1:10];
ridges=arrayfun(@(lambda) (XtTXt+lambda*I)\XtTYt,lambdas,"UniformOutput",false);
errs=cellfun(@(what) norm(wstar-what),ridges);
#errs=cellfun(@(what) norm(Xv*(what-wstar)),ridges);
losses=cellfun(@(what) norm(Xv*what-Yv),ridges);

semilogx(lambdas,errs,lambdas,losses,[lhat,lhat],[0,max([losses])])

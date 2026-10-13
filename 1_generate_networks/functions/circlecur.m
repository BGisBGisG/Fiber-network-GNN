function [X]=circlecur(O1,O2,d,f)
L=norm(O1-O2,2);
ns=L/d;
if ns>3
dp=floor((floor(L/d)+1)*1.5);
t=1:dp;
phi=rand().*2.*pi();
t=t.*2.*pi()./size(t,2);
r=f*L*cos((t-min(t))/max(t)*pi()-0.5*pi());
x=cos(t+phi).*r;
y=sin(t+phi).*r;
z=(1:size(t,2))./size(t,2).*L;
i=2;
X=zeros(100,3);ix=1;
op=[0,0,0];
while i<size(t,2)
    nowp=[x(1,i),y(1,i),z(1,i)];
    distances=norm(nowp-op,2);
    if distances>d
        op=nowp;
        X(ix,:)=op;
        ix=ix+1;
    end
    i=i+1;
end
X(ix:size(X,1),:)=[];
X=vectorrot(X,[0,0,1],O2-O1);
X=X+O1;
else
tt=(O2-O1)./4;
X=[tt.*1;tt.*2;tt.*3];
X=X+O1;
end
end

function [X1] = vectorrot(X, A, theta)
    D1 = A./norm(A,2);
    D2 = theta/norm(theta,2);
    if D1==D2
        X1=X;
    else
    A = cross(D1,D2);
    C = dot(D1,D2);
    S = sqrt(1-C^2);
A0 = A./norm(A,2);
Ax = A0(1); Ay = A0(2); Az = A0(3);
C1 = 1 - C;
M = ...
[
C1*Ax*Ax+C      C1*Ax*Ay-Az*S   C1*Ax*Az+Ay*S
C1*Ay*Ax+Az*S  C1*Ay*Ay+C       C1*Ay*Az-Ax*S
C1*Az*Ax-Ay*S   C1*Az*Ay+Ax*S  C1*Az*Az+C
];

X1 = (M*X')';
    end
end

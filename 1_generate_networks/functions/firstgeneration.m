function connectiontable=firstgeneration(X,B,maxconnection,minconnection)
if size(B,1)>1
big=sqrt(sum((B(:,2)-B(:,1)).^2,1)).*0.5;
else
    big=B;
end
connectiontable=cell(size(X,1),1);
for i=1:size(X,1)
connectiontable{i,1}=cell(1,3);
connectiontable{i,1}{1,1}=0;connectiontable{i,1}{1,2}=[];connectiontable{i,1}{1,3}=connectiontable{i,1}{1,2};
end
for i=1:size(X,1)
    point=X(i,:);
d=dis27(point,X,B);
range=d(:,2)<big;
Q=sortrows(d(range,:),2);
in=2;
while connectiontable{i,1}{1,1}<minconnection&&in<=size(X,1)
pointtry=Q(in,1);
if connectiontable{pointtry,1}{1,1}<maxconnection
if ismember(pointtry,connectiontable{i,1}{1,2})==0&&ismember(i,connectiontable{pointtry,1}{1,2})==0
    oldset=connectiontable{i,1}{1,2};
    connectiontable{i,1}{1,2}=[oldset,pointtry];
    oldset=connectiontable{i,1}{1,3};
    connectiontable{i,1}{1,3}=[oldset;Q(in,3:5)];
    connectiontable{i,1}{1,1}=connectiontable{i,1}{1,1}+1;

    oldset=connectiontable{pointtry,1}{1,2};
    connectiontable{pointtry,1}{1,2}=[oldset,i];
    oldset=connectiontable{pointtry,1}{1,3};
    connectiontable{pointtry,1}{1,3}=[oldset;-Q(in,3:5)];
    connectiontable{pointtry,1}{1,1}=connectiontable{pointtry,1}{1,1}+1;
end
end

in=in+1;
end
end
end

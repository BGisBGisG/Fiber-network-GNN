function fillconnectiontable=secondtgeneration(X,B,maxconnection,minconnection,connectiontable,range)
Bd=B;
if size(B,1)>1
    B=(B(:,2)-B(:,1))';
end
for i=1:size(connectiontable,1)
if connectiontable{i,1}{1,1}<maxconnection
connectionpoints=X(connectiontable{i,1}{1,2},:)+connectiontable{i,1}{1,3}.*B;
directingvectors=connectionpoints-X(i,:);
directingvectors=sum(directingvectors./sqrt(sum(directingvectors.^2,2)),1);
directingvectorslength=sqrt(sum(directingvectors.^2,2));
if directingvectorslength>0.5
directingvectors=-directingvectors;directingvectors=directingvectors./(sqrt(sum(directingvectors.^2,2))).*range.*0.5;
attachpoint=X(i,:)+directingvectors;breakflag=0;
while breakflag==0
distances=dis27(attachpoint,X,Bd);
useful=distances(:,2)<range;pointpool=distances(useful,:);
if ismember(i,pointpool(:,2))==1
loc=pointpool(:,2)==i;
pointpool(loc,:)=[];
end
yespool=[];
for k=1:size(pointpool,1)
if connectiontable{pointpool(k,1),1}{1,1}<maxconnection&&pointpool(k,1)~=i
yespool=[yespool,k];
end
end
pointpool=pointpool(yespool,:);

if isempty(pointpool)==0
    [~,lo]=min(pointpool(:,2),[],"all");
    pointtry=pointpool(lo,1);
        oldset=connectiontable{i,1}{1,2};
     connectiontable{i,1}{1,2}=[oldset,pointtry];
    oldset=connectiontable{i,1}{1,3};
    connectiontable{i,1}{1,3}=[oldset;pointpool(lo,3:5)];
    connectiontable{i,1}{1,1}=connectiontable{i,1}{1,1}+1;

    oldset=connectiontable{pointtry,1}{1,2};
    connectiontable{pointtry,1}{1,2}=[oldset,i];
    oldset=connectiontable{pointtry,1}{1,3};
    connectiontable{pointtry,1}{1,3}=[oldset;-pointpool(lo,3:5)];
    connectiontable{pointtry,1}{1,1}=connectiontable{pointtry,1}{1,1}+1;
    breakflag=1;
else
    attachpoint=attachpoint+directingvectors;
    if isin27cube(attachpoint,Bd)==0
        breakflag=1;
    end
end
end
end
end

end
fillconnectiontable=connectiontable;
end

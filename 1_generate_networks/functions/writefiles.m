function [P,K,A,Bd,avee]=writefiles(X,B,fillconnectiontable,curverate,littledistance,units)
if size(B,1)>1
big=floor(sqrt(sum((B(:,2)-B(:,1)).^2,1)).*0.5)*40;
Bb=(B(:,2)-B(:,1))';Bd=B;
else
    big=B;Bb=B;Bd=[0,B;0,B;0,B;];
end

connections=makeconnections(fillconnectiontable);
if size(littledistance,2)==1
littledistance=[1,littledistance];
end
P=zeros(size(X,1)*big,5);ip=1;P(:,1)=(1:size(X,1)*big)';P(:,2)=P(:,2)+littledistance(end,1);
P(ip:ip+size(X,1)-1,3:5)=X;ip=size(X,1)+1;
K=P(:,1:4);ik=1;A=P;ia=1;
atom_bond_angle_type=randi(size(littledistance,1), size(connections,1), 1);
for i=1:size(connections,1)
pointa=X(connections(i,1),:);pointb=X(connections(i,2),:)+connections(i,3:5).*Bb;
P2=circlecur(pointa,pointb,littledistance(atom_bond_angle_type(i,1),2),curverate);
arror=(ip:ip+size(P2,1)-1)';
P(arror,3:5)=P2;P(arror,2)=littledistance(atom_bond_angle_type(i,1),1);ip=ip+size(P2,1);
K1=[[connections(i,1);arror],[arror;connections(i,2)]];
K(ik:ik+size(P2,1),2)=littledistance(atom_bond_angle_type(i,1),1);
K(ik:ik+size(P2,1),3:4)=K1;ik=ik+size(P2,1)+1;
A1=[[connections(i,1);arror(1:size(arror,1)-1,:)],arror,[arror(2:size(arror,1),:);connections(i,2)]];
A(ia:ia+size(P2,1)-1,2)=littledistance(atom_bond_angle_type(i,1),1);
A(ia:ia+size(P2,1)-1,3:5)=A1;ia=ia+size(P2,1);
end
P(ip:size(P,1),:)=[];K(ik:size(K,1),:)=[];A(ia:size(A,1),:)=[];
P(:,3:5)=period(P(:,3:5),Bd);
P(:,3:5)=P(:,3:5).*units;P=[P(:,1),ones(size(P,1),1),P(:,2:5)];
Bd=Bd.*units;
avee=0;
for i=1:size(fillconnectiontable,1)
avee=avee+fillconnectiontable{i,1}{1,1};
end
avee=avee./size(fillconnectiontable,1);
if size(littledistance,2)==1
P(:,2)=1;K(:,2)=1;A(:,2)=1;
end
end

function connections=makeconnections(fillconnectiontable)
connections=zeros(size(fillconnectiontable,1)*10,5);icc=1;
for i=1:size(fillconnectiontable,1)
central=i;connectpoints=fillconnectiontable{central,1}{1,2};connectpointslocation=fillconnectiontable{central,1}{1,3};
for k=1:size(connectpointslocation,1)
checkpoint=connectpoints(1,k);checkpointvector=connectpointslocation(k,:);
zoness=fillconnectiontable{checkpoint,1}{1,2};
loca1=find(zoness==central);
if isempty(loca1)==0
checkpointlocation=fillconnectiontable{checkpoint,1}{1,3};
flag1=(sum((checkpointlocation(loca1,:)==-checkpointvector),2)==3);
else
    flag1=0;
    disp(append(num2str(central),' ',num2str(checkpoint)));
end

if flag1==1
connections(icc,:)=[central,checkpoint,checkpointvector];
icc=icc+1;
end
end
end
connections(icc:size(connections,1),:)=[];

yespool=1:size(connections,1);deletepool=[];
while isempty(yespool)==0
[nowcheck,l1]=min(yespool,[],"all");
a1=connections(nowcheck,1);a2=connections(nowcheck,2);
pa1=find(connections(:,1)==a2);pa2=find(connections(:,2)==a1);ppa=intersect(pa1,pa2);ppa=setdiff(ppa,nowcheck);
for o=1:size(ppa,1)
ppa2=find(yespool==ppa(o,1));
deletepool=[deletepool;ppa(o,1)];yespool(:,[l1;ppa2])=[];
end
end
connections(deletepool,:)=[];

end

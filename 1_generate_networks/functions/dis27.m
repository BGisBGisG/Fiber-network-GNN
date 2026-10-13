function [da]=dis27(p1,X,B)
ar=[-1	-1	-1
-1	-1	0
-1	-1	1
-1	0	-1
-1	0	0
-1	0	1
-1	1	-1
-1	1	0
-1	1	1
0	-1	-1
0	-1	0
0	-1	1
0	0	-1
0	0	0
0	0	1
0	1	-1
0	1	0
0	1	1
1	-1	-1
1	-1	0
1	-1	1
1	0	-1
1	0	0
1	0	1
1	1	-1
1	1	0
1	1	1];
if size(B,1)>1
B=(B(:,2)-B(:,1))';
end
da=zeros(size(ar,1)*size(X,1),5);
ll=(1:size(X,1))';lll=ones(size(X,1),3);l4=[ll,zeros(size(X,1),4)];
for i=1:size(ar,1)
    vector=ar(i,:);
    P=X+vector.*B;
    d1=sd2(p1,P);
    l4(:,2)=d1;
    l4(:,3:5)=lll.*vector;
    da((i-1)*size(X,1)+1:i*size(X,1),:)=l4;
end

end

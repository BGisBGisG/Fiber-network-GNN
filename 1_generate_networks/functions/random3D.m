function points = random3D(Bd, min_dist, num_points_requested,k)
cutsize=0.25;
reducefactor=0.1;
density = 0.74048;
    r = min_dist / 2;
    vol_sphere = (4/3) * pi * r^3;
    total_volume = prod((Bd(:,2)-Bd(:,1))');
    max_possible = floor(density * total_volume / vol_sphere);

    if max_possible<num_points_requested
        disp('fix to not set as ');
        while max_possible<num_points_requested
            min_dist=min_dist-0.01;
            r = min_dist / 2;
            vol_sphere = (4/3) * pi * r^3;
            max_possible = floor(density * total_volume / vol_sphere);
        end
        disp(num2str(min_dist));
        min_dist=min_dist*cutsize;
    end
rng(mod(floor(posixtime(datetime('now')) * 1e3),1e9)+floor(rand(1)*10));
flag=0;o=1;lsr=[1];gdd=[];
while o<k &&  flag==0
X=rand(num_points_requested,3).*(Bd(:,2)-Bd(:,1))';Xa=X;
pool=zeros(size(X,1),1);ip=1;c=1;
while c<50 && flag==0
    for i=1:size(X,1)
        D=dis76(X(i,:),X,[zeros(3,1),(Bd(:,2)-Bd(:,1))]);
        CC=setdiff(find(D<min_dist),i);
        if ~isempty(CC)
            pool(ip,1)=i;ip=ip+1;
        end
    end
lossrate=ip/size(X,1)*100;
lsr=[lsr,lossrate];
gd=lsr(1,size(lsr,2))-lsr(1,size(lsr,2)-1);
gdd=[gdd,gd];
disp(append('Reduced to',num2str(cutsize),' ',num2str(lossrate),'% is not good, try ',num2str(c),' times, restart ',num2str(o)));
if gdd(1,size(gdd,2))>=0
X=Xa;lsr(:,1:size(lsr,2)-1)=[];disp('back');
end
if size(gdd,2)>=15 && gdd(1,size(gdd,2))>=gdd(1,size(gdd,2)-14)
reducefactor=reducefactor/2;disp(append('reducefactor ',num2str(reducefactor)));
for u=1:14
gdd(1,size(gdd,2)-u)=gdd(1,size(gdd,2));
end
end
    if ip~=1
    Xa=X;
    X(pool(1:ip-1),:)=X(pool(1:ip-1),:)+rand(size(X(pool(1:ip-1),:),1),3).*(Bd(:,2)-Bd(:,1))'.*reducefactor;ip=1;
    X=period(X,[zeros(3,1),(Bd(:,2)-Bd(:,1))]);
    else
    flag=2;
    end
c=c+1;
end
o=o+1;
end
points=X+Bd(:,1)';
end

function [da]=dis76(p1,X,B)

if size(B,1)>1
    BB=[zeros(3,1),(B(:,2)-B(:,1))];X=X-B(:,1)';
B=(B(:,2)-B(:,1))';

else
    BB=[0,B;0,B;0,B;];
end

pt=B./2-p1;
X2=X+pt;
X2=period(X+pt,BB);
da=X2-B./2;
da=sqrt(sum(da.^2,2));

end

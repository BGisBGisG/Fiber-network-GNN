function [a]=period(b,c)
Bd=c;
a=b-Bd(:,1)';
cc=(Bd(:,2)-Bd(:,1))';
for i=1:3
c=cc(1,i);
a(:,i)=a(:,i)+ones(size(a,1),1).*c-floor((a(:,i)+ones(size(a,1),1).*c)./c).*c;
end
end

function []=writedata(name,keyword,atomt,bdt,agt,dit,Bd,Mass,X,B,A,D)
namen=name;
if size(name,2)<4
    name=append(name,'.txt');
else
    if strcmp(name(1,size(name,2)-3:size(name,2)),'.txt')==0
        name=append(name,'.txt');
    end
end

fid=fopen(name,"w+");

fprintf(fid,"%s\n",append(string(datetime),' ',keyword));
fprintf(fid,"\n");
if isempty(X)==0
    fprintf(fid,"%d atoms\n",size(X,1));
end
if isempty(B)==0
    fprintf(fid,"%d bonds\n",size(B,1));
end
if isempty(A)==0
    fprintf(fid,"%d angles\n",size(A,1));
end
if isempty(D)==0
    fprintf(fid,"%d dihedrals\n",size(D,1));
end
fprintf(fid,"\n");
fprintf(fid,"%d atom types\n",atomt);
fprintf(fid,"%d bond types\n",bdt);
fprintf(fid,"%d angle types\n",agt);
fprintf(fid,"%d dihedral types\n",dit);
fprintf(fid,"\n");
xc=['x';'y';'z'];
u1=kexuejishufa(Bd(:,1));u2=kexuejishufa(Bd(:,2));
for i=1:3
    st=append(num2str(u1(i,1)),'e',num2str(u1(i,2)),' ',num2str(u2(i,1)),'e',num2str(u2(i,2)),' ',xc(i,1),'lo',' ',xc(i,1),'hi');
    fprintf(fid,'%s\n',st);
end
fprintf(fid,"\n");
fprintf(fid,"Masses\n");
fprintf(fid,"\n");
m2=kexuejishufa(Mass(:,2));
for i=1:size(Mass,1)
    fprintf(fid,"%d %.10fe%d\n",Mass(i,1),m2(i,1),m2(i,2));
end
fclose(fid);

pp('Atoms',name,X);pp('Bonds',name,B);pp('Angles',name,A);pp('Dihedrals',name,D);

movefile(name,namen,"f");
end

function [output]=kexuejishufa(input)
if size(input,1)==1
    input=input';
end
number=input.*0;
digit=number;
nzp=find(input~=0);
digit(nzp,:)=floor(log10(abs(input(nzp,:))));
number(nzp,:)=input(nzp,:)./(10.^digit(nzp,:));
output=[number,digit];
end
function []=pp(modname,name,X)
if isempty(X)==0
fid=fopen(name,"a+");
st=append(newline,modname,newline,newline);
fprintf(fid,"%s",st);
fclose(fid);
writematrix(X,name,"FileType","text","Delimiter",' ','WriteMode','append');
end
end

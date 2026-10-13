% Junction counts of the networks to be generated. The heterogeneous dataset of the paper uses 50:100.
n=50:100;
% Number of networks (cases) generated for every junction count. The paper uses 200.
need=200;
% Fiber diameters in micrometers. The paper uses 0.05:0.01:0.20. The k-th value is diameter class k, which is atom, bond
% and angle type k in the LAMMPS input IN3 and uses the potential table fkT, so this list must stay as it is. To restrict
% the diameters, change dsel, the classes that are assigned at random to the fibers (dsel=6 gives 0.10 um for all fibers).
dlr=0.05:0.01:0.20;
dsel=1:numel(dlr);

runname='tier4';
dataname='voc2.data';
curverate=0.1;
B=[0 30;0 30;0 30;];
maxconnection=4;minconnection=3;
units=1e-6;

here=pwd;
addpath(fullfile(here,'functions'));
infile=fullfile(here,'..','2_simulation','IN3');
oprt=fullfile(here,'..','runs',runname);
if ~isfolder(oprt)
mkdir(oprt);
end

Mass=[1:size(dlr,2);(dlr.*units).^3./4.*pi.*1300e4]';
littledistancer=[1:size(dlr,2);dlr]';
littledistance=littledistancer(dsel,:);

fid=fopen(infile,'rb');
assert(fid>0,'No :%s',infile);
filesd=fread(fid,Inf,'*uint8');
fclose(fid);

warning('off','all');
nCores=feature('numcores');
nw=max(1,min(nCores-1,numel(n)*need));
p=parcluster('local');
p.NumWorkers=max(nCores,nw);
ppool=parpool(p,nw);

for N=n
parfor i=1:need
X=random3D(B,sqrt(sum((B(:,2)-B(:,1)).^2,1)).*0.5,N,1e5);finalN=size(X,1);
connectiontable=firstgeneration(X,B,maxconnection,minconnection);
fillconnectiontable=secondtgeneration(X,B,maxconnection,minconnection,connectiontable,sqrt(sum((B(:,2)-B(:,1)).^2,1)).*0.2);
[P,K,A,Bd,avee]=writefiles(X,B,fillconnectiontable,curverate,littledistance,units);
outd=fullfile(oprt,num2str(N*10000+i-1));
if ~isfolder(outd)
mkdir(outd);
end
writedata(fullfile(outd,dataname),append(' Average connection is ',num2str(avee),' ','n = ',num2str(finalN)),size(Mass,1),size(Mass,1),size(Mass,1),0,Bd,Mass,P,K,A,[]);
fid=fopen(fullfile(outd,'IN3'),'wb');
fwrite(fid,filesd,'uint8');
fclose(fid);
end
end
delete(ppool);
disp('GEN_DONE');

R=89e-9;dl=0.1e-6;Np=484;mu=0.5;sigma=0.08;a=2.16;
LJ_s=1;LJ_f=1;LJ_e=1;R_RBC=0.10;E_RBC=1e6;
if isfile("EI")
delete("EI");
end
i=1;
dll=(0.05:0.01:0.20).*1e-6;
for i=1:size(dll,2)
dl=dll(1,i);
[Kb] = fibrin_potentials(append('f',num2str(i)),R,dl,Np,mu,sigma,a,LJ_s,LJ_f,LJ_e,R_RBC,E_RBC);
disp(append('fix bk',num2str(i),' m bond/break 5 ',num2str(i),' ',num2str(dl*3),' prob 0.9 143'));
end
function [Kb] = fibrin_potentials(name,R,dl,Np,mu,sigma,a,LJ_s,LJ_f,LJ_e,R_RBC,E_RBC)
Densi=1300;

np0=0.0356e18;
Rs=R;
nps=(Np/(pi*R^2)-np0)*(a+2)/2+np0;
if nps >0
kp=1.34e-9;
lp=0.365e-9;
T=37+273.15;
nu=0.4;
kB = 1.38064852e-23;

stretching_potential = append(name,'T');
k1 = sqrt(2)*kB*T*Np^(3/2)/(lp*(mu+1));
k2 = Np*kp/(mu + 1);

length = linspace(dl/1e8,10*dl,1000000);
ep = (length - dl)/dl;

epprime = (1/2)*(ep + ...
                 sqrt(2*sigma^2/pi)*( exp(-(ep-mu).^2/(2*sigma^2)) - ...
                 exp(-mu^2/(2*sigma^2)) ) + ...
                 (ep-mu).*erf((ep-mu)/sqrt(2*sigma^2)) - ...
                 mu*erf(mu/sqrt(2*sigma^2)));

int_epprime = (1/4)*(ep.^2 + ...
                     sqrt(2*sigma^2/pi)*((ep-mu).*exp(-(ep-mu).^2/(2*sigma^2)) + ...
                     (mu-2*ep)*exp(-mu^2/(2*sigma^2))) + ...
                     ((ep-mu).^2 + sigma^2).*erf((ep-mu)/(sqrt(2*sigma^2))) + ...
                     (mu^2+sigma^2-2*mu*ep)*erf(mu/(sqrt(2*sigma^2))));

Fs = -(k1.*ep + (k2 - k1).*epprime);
Pis = dl*(k1*ep.^2/2 + (k2-k1).*int_epprime);

S=(0.5*dl).^2.*pi;sigmaa=-Fs./S;lll=abs(length-dl)./dl;Mou=diff(sigmaa)./diff(lll);
length=length./LJ_s;Pis=Pis./LJ_e;Fs=Fs.*LJ_s./LJ_e;
wm(stretching_potential,'STRETCH', [length;Pis;Fs]', 15, ' ');

Hup = [1, 1/2 + 5/a];
Hlo = (5+a)/a;

z = 2*(np0-nps)*(R/Rs)^a/((2+a)*np0);

H1 = real(hypergeom(Hup,Hlo,z));

Bs = 5*kB*T*pi*R^4*(2+a)*sqrt(pi*R^2*(np0 - 2*(np0-nps)*(R/Rs)^a/(2+a)))* ...
    (-2*(10+a)*(np0-nps)*(R/Rs)^a + np0*(20+a*(8+a)-(2/5)*a^2*H1))/ ...
    (2*sqrt(2)*(2+a)*(10+a)*(10+3*a)*lp*(1+mu));

Kb = 2*Bs/dl;

Kb=Kb./LJ_e;
fileID = fopen("EI",'a+');
fprintf(fileID,'angle_coeff %s %.15e 180\n', name(1,2:end),Kb);
fclose(fileID);
compression_potential =append(name, 'C');
R=dl/2;
ES_bar = sqrt(2)*kB*T*(pi*R^2*(np0-2*(np0-nps)*(R/Rs)^a/(2+a)))^(3/2)/(pi*R^2*lp*(mu+1));
Reff = R^2/(2*R);
Rij = linspace(1e-20*R,2*R,100000);
delta = 2*R - Rij;
ES_eff = (2*(1-nu^2)/(ES_bar))^-1;
Pic = 8*ES_eff*sqrt(Reff)*(2*R-Rij).^(5/2)/15;
Fc = (4/3)*ES_eff*sqrt(Reff)*delta.^(3/2);
Rij=Rij./LJ_s;Pic=Pic./LJ_e./1;Fc=Fc./LJ_f./1;
wm(compression_potential,'CONTACT', [Rij;Pic;Fc]', 15, ' ');
[Rij,Pic,Fc]=HertzContact(R_RBC,R_RBC,E_RBC,E_RBC);
compression_potential =append(name, 'RC');
[Rij,Pic,Fc]=HertzContact(R,R_RBC,ES_bar,E_RBC);
compression_potential = append(name,'fRC');
fclose('all');
else
disp('Surface density <0');
end
end

function [Rij,Pic,Fc]=HertzContact(R_A,R_B,E_A,E_B)
nu_fib = 1/2;
nu_RBC = 1/2;
R_fib=R_A;R_RBC=R_B;E_fib=E_A;E_RBC=E_B;
dij = R_fib + R_RBC;
Rij = 0.001*dij:(dij-0.001*dij)/(1000-1):dij;
Reff = (1/R_fib + 1/R_RBC)^(-1);
Eeff = ((1-nu_fib^2)/E_fib + (1-nu_RBC^2)/E_RBC)^(-1);
Pic = 8*(dij - Rij).^(5/2)*sqrt(Reff)*Eeff/15;
Fc = (4/3)*(dij - Rij).^(3/2)*sqrt(Reff)*Eeff;
end
function wm(filename,wd, A, precision, delimiter)
    [m, n] = size(A);
    fid = fopen(filename, 'w');
    fprintf(fid, ['# Lammps Description','\n\n']);
    fprintf(fid, [wd,'\n']);
fprintf(fid, ['N',' ',num2str(size(A,1)),'\n\n']);
    assert(fid > 0, 'Cannot open file: %s', filename);
    fmt_num = sprintf('%%.%de', precision);
    fmt = ['%d', delimiter,repmat([fmt_num, delimiter], 1, n-1), fmt_num, '\n' ];
    rowID = (1:m).';
    data  = [rowID, A];
    fprintf(fid, fmt, data.');
    fclose(fid);
end

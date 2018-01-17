%% C/P Structure, E-DrawDowns & Vulnerability in the World Stock-Market Network %%
%% Static Version %%
%% Input Lecture: %%

cd('C:\Users\Jairo F Gudiño R\Desktop\Stock Network Research') %Location of file%
[~,~,Database]=xlsread('MSCI Index.xlsx','Data'); %Lecture of the file%
%% Inputs and Parameters Definition: %%

Countries=Database(4,2:end);
Dates=Database(3538:end,1);
DaysWindow = 20;
CountriesNum=length(Countries);
n=4;
r=n+1;
permutations=100;
value=95;
Database=cell2mat(Database(3538:end,2:end));

Days = size(Database,1);
DatabaseW = Database;

%% Creation of Individual Draw-downs Vectors for each country: %%

Maxima=cell(Days-DaysWindow,CountriesNum);
Minima=cell(Days-DaysWindow,CountriesNum);
MaxIdx=cell(Days-DaysWindow,CountriesNum);
MinIdx=cell(Days-DaysWindow,CountriesNum);
DrawDowns=zeros(size(DatabaseW));

for d=DaysWindow:Days
DataWindow=DatabaseW(d-DaysWindow+1:d,:);
for y=1:CountriesNum
% Local extrema %
[MaximaX,MaxIdxX] = findpeaks(DataWindow(:,y));
Maxima{d-DaysWindow+1,y} = MaximaX;
MaxIdx{d-DaysWindow+1,y} = MaxIdxX;
[MinimaX, MinIdxX] = findpeaks(-DataWindow(:,y));
MinimaX = -MinimaX;
Minima{d-DaysWindow+1,y} = MinimaX;
MinIdx{d-DaysWindow+1,y} = MinIdxX;
% Correction Amplitude vs. E-drawdown.
if ~isempty(MaximaX) && ~isempty(MinimaX) && (length(MaximaX)>=2 && length(MinimaX)>=2)
p=1; q=1;
while p<=length(MaxIdxX)
MinimaX=MinimaX(MaxIdxX(p)<MinIdxX);
MinIdxX=MinIdxX(MaxIdxX(p)<MinIdxX);
if length(MinimaX)>1
E = MaximaX(p)-MinimaX(p+q);
CorrectionAmplitude=MaximaX(p+1)-MinimaX(p+q-1);
% Draw-down detection %
if E>CorrectionAmplitude
DrawDowns(MinIdxX(p+q)+d-DaysWindow,y)=1;
break
   else
p = p+1;
q = q-1;
end
   else
   break
end
end
    else
end

end
end
clear MinimaX MaximaX MaxIdxX MinIdxX MaxIdx MinIdx CorrectionAmplitude 
clear Maxima Minima p q g E
%% Matrices of Co-movements/Joint Drawdowns and Trend Reinforcement Indices %%

% Individual Probabilities %

DrawDays=size(DrawDowns,1);

% Individual Probabilities %
IndividualProb =(sum(DrawDowns)/Days);
IndividualProb = bsxfun(@times,IndividualProb,IndividualProb');

% Joint Probabilities %
% The effect of each i-row country on j-column country%
CountsCell=zeros(r,CountriesNum,CountriesNum,DrawDays-n);
CountsFirst=zeros(CountriesNum,CountriesNum,r);

for d=1:(DrawDays-n)
DataWindow=DrawDowns(d:d+n,:);
for y=1:CountriesNum
CountsCell(:,y,:,d) = DataWindow(1,y)*DataWindow;
end
end

for k=1:r
N = num2cell(permute(CountsCell(k,:,:,:),[2:r 1]),3);
CountsFirst(:,:,k)=arrayfun(@(x) sum(x{:}),N)/(DrawDays-n)-IndividualProb;
end
%% Correction for Randomness and Finite Size. Statistical Significance %%

MatrixOrder=zeros(DrawDays,permutations);
CountsPermMatrix=zeros(r,CountriesNum,CountriesNum,permutations);

for y=1:permutations
    
MatrixOrder(:,y)=(randperm(DrawDays))';
PermutationMatrix=(DrawDowns(MatrixOrder(:,y),:));
CountsPerm=zeros(r,CountriesNum,CountriesNum,DrawDays-n);
for d=1:(DrawDays-n)
DataWindow=PermutationMatrix(d:d+n,:);
for q=1:CountriesNum
CountsPerm(:,q,:,d) = DataWindow(1,q)*DataWindow;
end
end
for k=1:r
N = num2cell(permute(CountsPerm(k,:,:,:),[2:r 1]),3);
CountsPermMatrix(k,:,:,y)=arrayfun(@(x) sum(x{:}),N)/(DrawDays-n)-IndividualProb;
end

end

for k=1:r
N = num2cell(permute(CountsPermMatrix(k,:,:,:),[2 3 4 1]),3);
N = arrayfun(@(x) permute(x{:},[3 1 2]),N,'UniformOutput',0);
eval([strcat('CountsX',num2str(k)),'=CountsFirst(:,:,k).*(CountsFirst(:,:,k)>=arrayfun(@(x) prctile(x{:},value),N));'])
end

%% Interpreting the Conditional Probability Matrix W %%

% Interdependence Matrix and Trend Reinforcement Index %

FinalMatrix = (CountsX2+CountsX3+CountsX4)/3; % FinalMatrix = (CountsX3);
TrendReinforcement=FinalMatrix(logical(eye(size(FinalMatrix))));
FinalMatrix(logical(eye(size(FinalMatrix)))) = 0;

% Final Analysis: %

% Transmitters: %
InDegree=sum(FinalMatrix,2);
[~,Order]=sort(InDegree);
Transmitters=array2table(sort(InDegree),'RowNames',(Countries(Order))');

% Receptors: %
OutDegree=sum(FinalMatrix,1);
[~,Order]=sort(OutDegree);
Receptors=array2table(sort(OutDegree)','RowNames',(Countries(Order'))');

% Net: %
[~,Order]=sort(sum(FinalMatrix,2)-(sum(FinalMatrix,1))');
Net=array2table(sort(sum(FinalMatrix,2)-(sum(FinalMatrix,1))'),'RowNames',(Countries(Order))');
NetDegree=(sum(FinalMatrix,2)-(sum(FinalMatrix,1))');
% Trend Reinforcement Index: %
[~,Order]=sort(TrendReinforcement);
TrendReinforceH=array2table(sort(TrendReinforcement),'RowNames',(Countries(Order'))');

% Density Function: %
FinalDist=FinalMatrix(logical(FinalMatrix(:)));
histogram(FinalDist)

% Dates Analysis %
DrawDowns=array2table(DrawDowns,'RowNames',Dates,'VariableNames',Countries);
x=table2array(DrawDowns);
x=array2table(sum(x,2),'RowNames',Dates);
DrawDowns=table2array(DrawDowns);

clear Order;

%% Impacting and Impacted Centrality %%

% Initial values for authority and hub scores of pages:
t = 1E-3;
a = ones(CountriesNum, 1) / sqrt(CountriesNum); % Authority score
k = ones(CountriesNum, 1) / sqrt(CountriesNum); % Hub score
% Keep list of values for authority and hub scores during the iteration:
aValues = zeros(CountriesNum, 0);
kValues = zeros(CountriesNum, 0);
numIter = 0;
while 1
    % Old authority and hub scores:
    aOld = a; 
    kOld = k;
    % Update authority and hub scores and scale to unity:
    a = FinalMatrix' * kOld / sqrt(sum((FinalMatrix' * kOld).^2)); 
    k = FinalMatrix * aOld / sqrt(sum((FinalMatrix * aOld).^2));
    aDiff = abs(a - aOld);
    kDiff = abs(k - kOld);
    % Append new authority and hub scores to lists:
    aValues(:, end + 1) = a;
    kValues(:, end + 1) = k;
    if all(aDiff < t) && all(kDiff < t)
        break;
    end
   numIter = numIter + 1;
end

hub_ranks = (k/sum(k));
[~,Order]=sort(hub_ranks);
hub_ranksH=array2table(sort(hub_ranks),'RowNames',(Countries(Order'))');

auth_ranks = (a/sum(a));
[~,Order]=sort(auth_ranks);
auth_ranksH=array2table(sort(auth_ranks),'RowNames',(Countries(Order'))');

% % Scatter Analysis %
% scatter(OutDegree',auth_ranks);
% scatter(InDegree,hub_ranks);
% scatter(TrendReinforcement,OutDegree');

%% C/P Structure %%

Core = PublicUseCore(FinalMatrix);
CoreCountries = Countries(logical(Core'));
PeripheryCountries = Countries(logical(~Core'));

%% Printing of Matrices %%


% save ('hub_ranks','hub_ranks')
% save ('auth_ranks','auth_ranks')
% save ('CountsX1','CountsX1')
% save ('CountsX2','CountsX1')
% save ('CountsX3','CountsX3')
% save ('CountsX4','CountsX4')
% save ('CountsX5','CountsX5')
% save ('FinalMatrix','FinalMatrix')
% save ('DrawDowns','DrawDowns')

%% Controllability %%

[Nd,~,~] = ExactControllability(FinalMatrix,'plotting',0);
% Sólo hay una configuración del driver node.
% 1 sólo driver node.

%% End of the Code %%

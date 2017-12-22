%% Bow-Tie Structure, E-DrawDowns & Vulnerability in the World Stock-Market Network %%
%% Input Lecture: %%

cd('C:\Users\Jairo F Gudiño R\Desktop\Stock Network Research')
[~,~,Database]=xlsread('MSCI Index.xlsx','Data');
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

FinalMatrix = (CountsX2+CountsX3+CountsX4)/3;
% FinalMatrix = (CountsX3);
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

% % Dates Analysis %
% DrawDowns=array2table(DrawDowns,'RowNames',Dates,'VariableNames',Countries);
% x=table2array(DrawDowns);
% x=array2table(sum(x,2),'RowNames',Dates);
% DrawDowns=table2array(DrawDowns);

clear Order;

%% Impacting and Impacted Centrality %%

[r,c,v] = find(FinalMatrix);
AdjacencyL = [r,c,v];

G = digraph(AdjacencyL(:,1),AdjacencyL(:,2),AdjacencyL(:,3));
G.Nodes.Name=Countries';
% plot(G)

% PageRank Algorithm %
pg_ranksR = centrality(G,'pagerank');
[~,Order]=sort(pg_ranksR);
PageRankListReceptor=array2table(sort(pg_ranks),'RowNames',(Countries(Order'))');

% Hub & Authority Algorithms %
hub_ranks = centrality(G,'hubs');
[~,Order]=sort(hub_ranks);
HubList=array2table(sort(hub_ranks),'RowNames',(Countries(Order'))');

auth_ranks = centrality(G,'authorities');
[~,Order]=sort(auth_ranks);
AuthorityList=array2table(sort(auth_ranks),'RowNames',(Countries(Order'))');

% scatter(hub_ranks,auth_ranks)

[~,Order]=sort(hub_ranks-auth_ranks);
BowTieListHA=array2table(sort(hub_ranks-auth_ranks),'RowNames',(Countries(Order'))');

G.Nodes.Hubs = hub_ranks;
G.Nodes.Authorities = auth_ranks;

[r,c,v] = find(FinalMatrix');
AdjacencyL = [r,c,v];

G = digraph(AdjacencyL(:,1),AdjacencyL(:,2),AdjacencyL(:,3));
G.Nodes.Name=Countries';
% plot(G)

% PageRank Algorithm %
pg_ranksT = centrality(G,'pagerank');
[~,Order]=sort(pg_ranksT);
PageRankListTransmitter=array2table(sort(pg_ranks),'RowNames',(Countries(Order'))');

[~,Order]=sort(pg_ranksT-pg_ranksR);
BowTieListPR=array2table(sort(pg_ranksT-pg_ranksR),'RowNames',(Countries(Order'))');

% scatter(pg_ranksT,pg_ranksR)

%% Controllability %%

[Nd,~,~] = ExactControllability(FinalMatrix,'plotting',1);
% Sólo hay una configuración del driver node.
% 1 sólo driver node.

%% Save Results %%

save PageRankListTransmitter
save BowTieListPR
save BowTieListHA

save pg_ranksR
save pg_ranksT
save hub_ranks
save auth_ranks

save PageRankListReceptor
save HubList
save AuthorityList

save CountsX1
save CountsX2
save CountsX3
save CountsX4
save CountsX5
save FinalMatrix
save DrawDowns

%% Tiered structure %%

% Core = PublicUseCore(FinalMatrix);
% CoreCountries = Countries(logical(Core'));
% PeripheryCountries = Countries(logical(~Core'));

% % C-Fuzzy Means: %
% s = rng(2)
% options = [NaN 25 0.001 0];
% [centers,U] = fcm(NetDegree,3,options);
% [~,b]=sort(centers,'descend');
% maxU = max(U);
% Core = Countries(find(U(b(1),:) == maxU));
% Center = Countries(find(U(b(2),:) == maxU));
% Periphery = Countries(find(U(b(3),:) == maxU));

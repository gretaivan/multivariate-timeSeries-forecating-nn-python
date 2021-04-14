%%Connect to the FRED data server 
url = 'https://fred.stlouisfed.org/';
c = fred(url);


%set format with numerical date 
format bank;

c.DataReturnFormat = 'table';
c.DatetimeType = 'datetime';

%get data with date filter to match USD/EUR rates
startdate = '04/01/1999'; % beginning of date range for historical data
enddate = '04/01/2020'; % ending of date range for historical data

%%
%%U.S. Dollars to One Euro, Daily, Not Seasonally Adjusted

usdEur = 'DEXUSEU'; 
usdEur = fetch(c,usdEur,startdate,enddate);

%usdEur.Data(1:3,:)
%usdEur = usdEur.Data{1};
%head(usdEur)

%%
%%NASDAQ composite index (all companies) daily, not seasonally adjusted
NASDAQ = 'NASDAQCOM';

NASDAQ = fetch(c,NASDAQ,startdate,enddate);
%NASDAQ.Data(1:3,:);
%NASDAQ = NASDAQ.Data{1};

%%
%%NASDAQ 100 index (top 100 companies) daily, not seasonally adjusted
NASDAQ100 = 'NASDAQ100';

NASDAQ100 = fetch(c,NASDAQ100,startdate,enddate);
%NASDAQ100.Data(1:3,:)
%NASDAQ100 = NASDAQ100.Data{1};

%%
%%GOLD USD
GoldUsd = 'GOLDAMGBD228NLBM';
GoldUsd = fetch(c, GoldUsd, startdate, enddate);
%GoldUsdLondon.Data(1:3,:);
%GoldUsd = GoldUsd.Data{1}


%%
%%Global WTI 

WTI = 'DCOILWTICO';
WTI = fetch(c, WTI, startdate, enddate);
%WTI = WTI.Data{1}
close(c)


%%
%%review data
%WTI.Data(1:3,:);

%%
%%FRANKFURT

%bloomberg = blp;
%s = 'DAX:IND';
%f = {'LAST_PRICE';'OPEN'}; 
%period = 'daily'; 

%[d,sec] = history(bloomberg,s,f,startdate,enddate,period)


%%CALL getData.m file to extract the data from FRED
getData

%%
%convert 5 variables to the suitable format
%NOTE: file converter.m is needed! as it is custom funtion

usd = converter(usdEur)
wti = converter(WTI)
xau = converter(GoldUsd)
nasdaq = converter(NASDAQ)
nasdaq100 = converter(NASDAQ100)

%%
%EXPORT TO CSV
writetable(usd, 'usd.csv')
writetable(wti, 'wti.csv')
writetable(xau,'xau.csv')
writetable(nasdaq,'nasdaq.csv')
writetable(nasdaq100,'nasdaq100.csv')

disp("Export completed")


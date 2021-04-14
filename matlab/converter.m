function file = converter(x)
    x = x.Data{1};
    xTime = x.Var1;
    xTime.Format = 'dd/MM/yyyy'
    
    file = table(xTime, x.Var2)
end
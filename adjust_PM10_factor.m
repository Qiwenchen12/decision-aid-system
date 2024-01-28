function PM10_factor = adjust_PM10_factor(PM10_factor)
for i = 1:numel(PM10_factor)
     if PM10_factor(i)>= 250 && PM10_factor(i)< 350
            PM10_factor(i) = 200;
        elseif PM10_factor(i) >= 150 && PM10_factor(i) < 250
            PM10_factor(i) = 100;
        elseif PM10_factor(i)  > 50 &&  PM10_factor(i) < 150
            PM10_factor(i)  = 50;
        elseif PM10_factor(i)<= 50
         if i < numel(PM10_factor)  
                continue; 
            else   
            return;
        end
    end
end
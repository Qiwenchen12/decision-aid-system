function repair_factor = adjust_repair_factor(repair_factor)
for i = 1:numel(repair_factor)
     if repair_factor(i) > 60 &&repair_factor(i)< 70
            repair_factor(i) = 75;
        elseif repair_factor(i) >= 70 && repair_factor(i)< 80
            repair_factor(i) = 85;
        elseif repair_factor(i)  >= 80 &&  repair_factor(i)< 90
            repair_factor(i)  = 90;
        elseif repair_factor(i)>= 90
        if i < numel(repair_factor)  
                continue; 
            else  
            return;
        end
    end
end
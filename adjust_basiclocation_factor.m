function basiclocation_factor = adjust_basiclocation_factor(basiclocation_factor)
 for i = 1:numel(basiclocation_factor)
    if basiclocation_factor(i)> 60 && basiclocation_factor(i)< 70
            basiclocation_factor(i)= 75;
        elseif basiclocation_factor(i) >= 70 &&basiclocation_factor(i)  < 80
            basiclocation_factor(i)  = 85;
        elseif basiclocation_factor(i)  >= 80 &&  basiclocation_factor(i)< 90
            basiclocation_factor(i) = 90;
        elseif basiclocation_factor(i)>= 90
         if i < numel(basiclocation_factor)  
                continue; 
            else
            return;
        end
    end
end
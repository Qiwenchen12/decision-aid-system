function management_factor = adjust_management_factor(management_factor)
for i = 1:numel(management_factor)
     if management_factor(i)  > 60 && management_factor(i)< 70
            management_factor(i)  = 75;
        elseif management_factor(i) >= 70 && management_factor(i)< 80
            management_factor(i) = 85;
        elseif management_factor(i)>= 80 &&management_factor(i) < 90
            management_factor(i) = 90;
        elseif management_factor(i)>= 90
         if i < numel(management_factor)  
                continue; 
            else   
            return;
        end
    end
end
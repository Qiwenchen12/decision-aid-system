function drill_factor = adjust_drill_factor(drill_factor)
   for i = 1:numel(drill_factor)
        if drill_factor(i)  > 6 &&drill_factor(i)<= 12
            drill_factor(i) = 18;
        elseif drill_factor(i) > 12 && drill_factor(i)<= 24
            drill_factor(i)  = 30;
        elseif drill_factor(i)> 24 && drill_factor(i) <= 36
            drill_factor(i) = 36;
        elseif drill_factor(i)> 36
         if i < numel(drill_factor)  
                continue; 
            else   
            return;
        end
    end
end

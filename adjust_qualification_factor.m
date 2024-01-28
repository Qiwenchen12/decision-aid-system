function qualification_factor = adjust_qualification_factor(qualification_factor)
for i = 1:numel(qualification_factor)
     if qualification_factor(i) > 80 && qualification_factor(i)< 85
            qualification_factor(i) = 87.5;
        elseif qualification_factor(i) >= 85&& qualification_factor(i)  < 90
            qualification_factor(i) = 92.5;
        elseif qualification_factor(i)  >= 90 && qualification_factor(i) < 95
            qualification_factor(i) = 95;
        elseif qualification_factor(i)>= 95
            if i < numel(qualification_factor)  
                continue; 
            else
            return;
        end
    end
end
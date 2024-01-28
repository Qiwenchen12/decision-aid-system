function safety_factor = adjust_safety_factor(safety_factor)
for i = 1:numel(safety_factor)
    if safety_factor(i) > 12 &&  safety_factor(i)<= 24
            safety_factor(i) = 36;
        elseif safety_factor(i)  > 24 && safety_factor(i)<= 48
            safety_factor(i)  = 72;
        elseif safety_factor(i) > 48 && safety_factor(i)<= 96
            safety_factor(i)= 96;
        elseif safety_factor(i)> 96
         if i < numel(safety_factor)  
                continue; 
            else
            return;
        end
    end
end
function inspection_factor = adjust_inspection_factor(inspection_factor)
for i = 1:numel(inspection_factor)
    if inspection_factor(i) == 0.4 
            inspection_factor(i) = 0.6;
        elseif inspection_factor(i) == 0.6
            inspection_factor(i) = 0.8;
        elseif inspection_factor(i) == 0.8
            inspection_factor(i) = 1.0;
        elseif inspection_factor(i) == 1.0
         if i < numel(inspection_factor)  
                continue; 
            else   
            return;
        end
    end
end

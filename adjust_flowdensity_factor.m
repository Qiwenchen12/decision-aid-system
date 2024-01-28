function flowdensity_factor = adjust_flowdensity_factor(flowdensity_factor)
for i = 1:numel(flowdensity_factor)
     if flowdensity_factor(i)>= 3.26 &&flowdensity_factor(i)< 4.18
            flowdensity_factor(i) = 2.8;
        elseif flowdensity_factor(i)  >= 2.34 && flowdensity_factor(i)  < 3.26
            flowdensity_factor(i) = 1.88;
        elseif flowdensity_factor(i) > 1.42 && flowdensity_factor(i) < 2.34
            flowdensity_factor(i)  = 1.42;
        elseif flowdensity_factor(i)<= 1.42
         if i < numel(flowdensity_factor)  
                continue; 
            else  
            return;
        end
    end
end
function risk_factor = adjust_risk_factor(risk_factor)
    for i = 1:numel(risk_factor)
        if risk_factor(i) == 0.4
            risk_factor(i) = 0.6;
        elseif risk_factor(i) == 0.6
            risk_factor(i) = 0.8;
        elseif risk_factor(i) == 0.8
            risk_factor(i) = 1.0;
        elseif risk_factor(i) == 1.0
            if i < numel(risk_factor)  
                continue; 
            else
                return; 
            end
        end
    end
end

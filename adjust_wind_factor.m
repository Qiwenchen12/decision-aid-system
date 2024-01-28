function wind_factor = adjust_wind_factor(wind_factor)
     for i = 1:numel(wind_factor)
        if wind_factor(i) >= 0.5 && wind_factor(i) < 1
            wind_factor(i) = 0.4;
        elseif wind_factor(i) >= 0.3 &&  wind_factor(i)< 0.5
            wind_factor(i) = 0.225;
        elseif wind_factor(i) > 0.15 && wind_factor(i) < 0.3
            wind_factor(i) = 0.15;
        elseif wind_factor(i) <= 0.15
           
            return;
        end
     end

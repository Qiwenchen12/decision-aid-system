% 提取列的数据
light_factor = table11.light_factor;
temperature_factor = table11.temperature_factor;
humidity_factor = table11.humidity_factor;
wind_factor = table11.wind_factor;
noise_factor = table11.noise_factor;
PM10_factor = table11.PM10_factor;
PM25_factor = table11.PM25_factor;
CO2_factor = table11.CO2_factor;
CO_factor = table11.CO_factor;
TVOC_factor = table11.TVOC_factor;
flowdensity_factor = table11.flowdensity_factor;
natural_factor = table11.natural_factor;
social_factor = table11.social_factor;
employee_factor = table11.employee_factor;
technical_factor = table11.technical_factor;
skill_factor = table11.skill_factor;
basiclocation_factor = table11.basiclocation_factor;
emergencylocation_factor = table11.emergencylocation_factor;
basicalconfiguration_factor = table11.basicalconfiguration_factor;
emergencyconfiguration_factor = table11.emergencyconfiguration_factor;
repair_factor = table11.repair_factor;
qualification_factor = table11.qualification_factor;
safety_factor = table11.safety_factor;
drill_factor = table11.drill_factor;
management_factor = table11.management_factor;
plan_factor = table11.plan_factor;
inspection_factor = table11.inspection_factor;
risk_factor = table11.risk_factor;
organization_factor = table11.organization_factor;
bacteria_factor = table11.bacteria_factor;


% 调用函数计算调整列
social_factor = adjust_social_factor(social_factor);
plan_factor= adjust_plan_factor(plan_factor);
employee_factor =adjust_employee_factor (employee_factor);
PM10_factor= adjust_PM10_factor(PM10_factor);
management_factor= adjust_management_factor(management_factor);
bacteria_factor= adjust_bacteria_factor(bacteria_factor);
PM25_factor= adjust_PM25_factor (PM25_factor);
inspection_factor = adjust_inspection_factor(inspection_factor);
CO_factor= adjust_CO_factor(CO_factor);
technical_factor= adjust_technical_factor(technical_factor);
qualification_factor= adjust_qualification_factor(qualification_factor);
light_factor= adjust_light_factor(light_factor);
drill_factor= adjust_drill_factor(drill_factor);
CO2_factor = adjust_CO2_factor(CO2_factor);
risk_factor = adjust_risk_factor(risk_factor);
emergencyconfiguration_factor= adjust_emergencyconfiguration_factor(emergencyconfiguration_factor);
basiclocation_factor= adjust_basiclocation_factor(basiclocation_factor);
emergencylocation_factor= adjust_emergencylocation_factor(emergencylocation_factor);
noise_factor= adjust_noise_factor(noise_factor);
natural_factor =adjust_natural_factor (natural_factor);
TVOC_factor = adjust_TVOC_factor(TVOC_factor);
organization_factor= adjust_organization_factor(organization_factor);
skill_factor= adjust_skill_factor(skill_factor);
safety_factor= adjust_safety_factor(safety_factor);
repair_factor= adjust_repair_factor(repair_factor);
humidity_factor= adjust_humidity_factor(humidity_factor);
basicalconfiguration_factor= adjust_basicalconfiguration_factor(basicalconfiguration_factor);
flowdensity_factor= adjust_flowdensity_factor(flowdensity_factor);
temperature_factor= adjust_temperature_factor(temperature_factor);
wind_factor= wind_factor;

% 创建一个新表格
adjustedTable1 = table11;
adjustedTable1.employee_factor = employee_factor;

adjustedTable2 = table11;
adjustedTable2.safety_factor = safety_factor;

adjustedTable3 = table11;
adjustedTable3.PM25_factor = PM25_factor;

adjustedTable4 = table11;
adjustedTable4.PM10_factor = PM10_factor;

adjustedTable5 = table11;
adjustedTable5.emergencyconfiguration_factor = emergencyconfiguration_factor;

adjustedTable6 = table11;
adjustedTable6.bacteria_factor = bacteria_factor;

adjustedTable7 = table11;
adjustedTable7.drill_factor = drill_factor;

adjustedTable8 = table11;
adjustedTable8.organization_factor = organization_factor;

adjustedTable9 = table11;
adjustedTable9.qualification_factor = qualification_factor;

adjustedTable10 = table11;
adjustedTable10.technical_factor = technical_factor;

adjustedTable11 = table11;
adjustedTable11.plan_factor = plan_factor;

adjustedTable12 = table11;
adjustedTable12.CO2_factor = CO2_factor;

adjustedTable13 = table11;
adjustedTable13.repair_factor = repair_factor;

adjustedTable14 = table11;
adjustedTable14.social_factor = social_factor;

adjustedTable15 = table11;
adjustedTable15.basiclocation_factor = basiclocation_factor;

adjustedTable16 = table11;
adjustedTable16.skill_factor = skill_factor;

adjustedTable17 = table11;
adjustedTable17.temperature_factor = temperature_factor;

adjustedTable18 = table11;
adjustedTable18.natural_factor = natural_factor;

adjustedTable19 = table11;
adjustedTable19.flowdensity_factor = flowdensity_factor;

adjustedTable20 = table11;
adjustedTable20.management_factor = management_factor;

adjustedTable21 = table11;
adjustedTable21.light_factor = light_factor;

adjustedTable22 = table11;
adjustedTable22.CO_factor = CO_factor;

adjustedTable23 = table11;
adjustedTable23.basicalconfiguration_factor = basicalconfiguration_factor;

adjustedTable24 = table11;
adjustedTable24.inspection_factor = inspection_factor;

adjustedTable25 = table11;
adjustedTable25.risk_factor = risk_factor;

adjustedTable26 = table11;
adjustedTable26.noise_factor = noise_factor;

adjustedTable27 = table11;
adjustedTable27.TVOC_factor = TVOC_factor;

adjustedTable28 = table11;
adjustedTable28.wind_factor = wind_factor;

adjustedTable29 = table11;
adjustedTable29.humidity_factor = humidity_factor;

adjustedTable30 = table11;
adjustedTable30.emergencylocation_factor = emergencylocation_factor;



%放在同一表下
table12 = vertcat(adjustedTable1, adjustedTable2,adjustedTable3,adjustedTable4,adjustedTable5,adjustedTable6,adjustedTable7,adjustedTable8,adjustedTable9,adjustedTable10,adjustedTable11, adjustedTable12,adjustedTable13,adjustedTable14,adjustedTable15,adjustedTable16,adjustedTable17,adjustedTable18,adjustedTable19,adjustedTable20,adjustedTable21, adjustedTable22,adjustedTable23,adjustedTable24,adjustedTable25,adjustedTable26,adjustedTable27,adjustedTable28,adjustedTable29,adjustedTable30);


%删除重复项
[~, uniqueIndices] = unique(table12, 'rows', 'stable');
table12Unique = table12(uniqueIndices, :);

rows_table11Unique_noDup = table11{:,:};
rows_table12Unique = table12Unique{:,:};
[~, idx1] = ismember(rows_table12Unique, rows_table11Unique_noDup, 'rows');
table12 = table12Unique(~idx1, :);


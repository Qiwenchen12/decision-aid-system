% 提取列的数据
light_factor = table5.light_factor;
temperature_factor = table5.temperature_factor;
humidity_factor = table5.humidity_factor;
wind_factor = table5.wind_factor;
noise_factor = table5.noise_factor;
PM10_factor = table5.PM10_factor;
PM25_factor = table5.PM25_factor;
CO2_factor = table5.CO2_factor;
CO_factor = table5.CO_factor;
TVOC_factor = table5.TVOC_factor;
flowdensity_factor = table5.flowdensity_factor;
natural_factor = table5.natural_factor;
social_factor = table5.social_factor;
employee_factor = table5.employee_factor;
technical_factor = table5.technical_factor;
skill_factor = table5.skill_factor;
basiclocation_factor = table5.basiclocation_factor;
emergencylocation_factor = table5.emergencylocation_factor;
basicalconfiguration_factor = table5.basicalconfiguration_factor;
emergencyconfiguration_factor = table5.emergencyconfiguration_factor;
repair_factor = table5.repair_factor;
qualification_factor = table5.qualification_factor;
safety_factor = table5.safety_factor;
drill_factor = table5.drill_factor;
management_factor = table5.management_factor;
plan_factor = table5.plan_factor;
inspection_factor = table5.inspection_factor;
risk_factor = table5.risk_factor;
organization_factor = table5.organization_factor;
bacteria_factor = table5.bacteria_factor;




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
adjustedTable1 = table5;
adjustedTable1.employee_factor = employee_factor;

adjustedTable2 = table5;
adjustedTable2.safety_factor = safety_factor;

adjustedTable3 = table5;
adjustedTable3.PM25_factor = PM25_factor;

adjustedTable4 = table5;
adjustedTable4.PM10_factor = PM10_factor;

adjustedTable5 = table5;
adjustedTable5.emergencyconfiguration_factor = emergencyconfiguration_factor;

adjustedTable6 = table5;
adjustedTable6.bacteria_factor = bacteria_factor;

adjustedTable7 = table5;
adjustedTable7.drill_factor = drill_factor;

adjustedTable8 = table5;
adjustedTable8.organization_factor = organization_factor;

adjustedTable9 = table5;
adjustedTable9.qualification_factor = qualification_factor;

adjustedTable10 = table5;
adjustedTable10.technical_factor = technical_factor;

adjustedTable11 = table5;
adjustedTable11.plan_factor = plan_factor;

adjustedTable12 = table5;
adjustedTable12.CO2_factor = CO2_factor;

adjustedTable13 = table5;
adjustedTable13.repair_factor = repair_factor;

adjustedTable14 = table5;
adjustedTable14.social_factor = social_factor;

adjustedTable15 = table5;
adjustedTable15.basiclocation_factor = basiclocation_factor;

adjustedTable16 = table5;
adjustedTable16.skill_factor = skill_factor;

adjustedTable17 = table5;
adjustedTable17.temperature_factor = temperature_factor;

adjustedTable18 = table5;
adjustedTable18.natural_factor = natural_factor;

adjustedTable19 = table5;
adjustedTable19.flowdensity_factor = flowdensity_factor;

adjustedTable20 = table5;
adjustedTable20.management_factor = management_factor;

adjustedTable21 = table5;
adjustedTable21.light_factor = light_factor;

adjustedTable22 = table5;
adjustedTable22.CO_factor = CO_factor;

adjustedTable23 = table5;
adjustedTable23.basicalconfiguration_factor = basicalconfiguration_factor;

adjustedTable24 = table5;
adjustedTable24.inspection_factor = inspection_factor;

adjustedTable25 = table5;
adjustedTable25.risk_factor = risk_factor;

adjustedTable26 = table5;
adjustedTable26.noise_factor = noise_factor;

adjustedTable27 = table5;
adjustedTable27.TVOC_factor = TVOC_factor;

adjustedTable28 = table5;
adjustedTable28.wind_factor = wind_factor;

adjustedTable29 = table5;
adjustedTable29.humidity_factor = humidity_factor;

adjustedTable30 = table5;
adjustedTable30.emergencylocation_factor = emergencylocation_factor;



%放在同一表下
table6 = vertcat(adjustedTable1, adjustedTable2,adjustedTable3,adjustedTable4,adjustedTable5,adjustedTable6,adjustedTable7,adjustedTable8,adjustedTable9,adjustedTable10,adjustedTable11, adjustedTable12,adjustedTable13,adjustedTable14,adjustedTable15,adjustedTable16,adjustedTable17,adjustedTable18,adjustedTable19,adjustedTable20,adjustedTable21, adjustedTable22,adjustedTable23,adjustedTable24,adjustedTable25,adjustedTable26,adjustedTable27,adjustedTable28,adjustedTable29,adjustedTable30);

%删除重复项
[~, uniqueIndices] = unique(table6, 'rows', 'stable');
table6Unique = table6(uniqueIndices, :);

rows_table5Unique_noDup = table5{:,:};
rows_table6Unique = table6Unique{:,:};
[~, idx1] = ismember(rows_table6Unique, rows_table5Unique_noDup, 'rows');
table6 = table6Unique(~idx1, :);


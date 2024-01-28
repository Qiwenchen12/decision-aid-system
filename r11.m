% 提取列的数据
light_factor = table10.light_factor;
temperature_factor = table10.temperature_factor;
humidity_factor = table10.humidity_factor;
wind_factor = table10.wind_factor;
noise_factor = table10.noise_factor;
PM10_factor = table10.PM10_factor;
PM25_factor = table10.PM25_factor;
CO2_factor = table10.CO2_factor;
CO_factor = table10.CO_factor;
TVOC_factor = table10.TVOC_factor;
flowdensity_factor = table10.flowdensity_factor;
natural_factor = table10.natural_factor;
social_factor = table10.social_factor;
employee_factor = table10.employee_factor;
technical_factor = table10.technical_factor;
skill_factor = table10.skill_factor;
basiclocation_factor = table10.basiclocation_factor;
emergencylocation_factor = table10.emergencylocation_factor;
basicalconfiguration_factor = table10.basicalconfiguration_factor;
emergencyconfiguration_factor = table10.emergencyconfiguration_factor;
repair_factor = table10.repair_factor;
qualification_factor = table10.qualification_factor;
safety_factor = table10.safety_factor;
drill_factor = table10.drill_factor;
management_factor = table10.management_factor;
plan_factor = table10.plan_factor;
inspection_factor = table10.inspection_factor;
risk_factor = table10.risk_factor;
organization_factor = table10.organization_factor;
bacteria_factor = table10.bacteria_factor;


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
adjustedTable1 = table10;
adjustedTable1.employee_factor = employee_factor;

adjustedTable2 = table10;
adjustedTable2.safety_factor = safety_factor;

adjustedTable3 = table10;
adjustedTable3.PM25_factor = PM25_factor;

adjustedTable4 = table10;
adjustedTable4.PM10_factor = PM10_factor;

adjustedTable5 = table10;
adjustedTable5.emergencyconfiguration_factor = emergencyconfiguration_factor;

adjustedTable6 = table10;
adjustedTable6.bacteria_factor = bacteria_factor;

adjustedTable7 = table10;
adjustedTable7.drill_factor = drill_factor;

adjustedTable8 = table10;
adjustedTable8.organization_factor = organization_factor;

adjustedTable9 = table10;
adjustedTable9.qualification_factor = qualification_factor;

adjustedTable10 = table10;
adjustedTable10.technical_factor = technical_factor;

adjustedTable11 = table10;
adjustedTable11.plan_factor = plan_factor;

adjustedTable12 = table10;
adjustedTable12.CO2_factor = CO2_factor;

adjustedTable13 = table10;
adjustedTable13.repair_factor = repair_factor;

adjustedTable14 = table10;
adjustedTable14.social_factor = social_factor;

adjustedTable15 = table10;
adjustedTable15.basiclocation_factor = basiclocation_factor;

adjustedTable16 = table10;
adjustedTable16.skill_factor = skill_factor;

adjustedTable17 = table10;
adjustedTable17.temperature_factor = temperature_factor;

adjustedTable18 = table10;
adjustedTable18.natural_factor = natural_factor;

adjustedTable19 = table10;
adjustedTable19.flowdensity_factor = flowdensity_factor;

adjustedTable20 = table10;
adjustedTable20.management_factor = management_factor;

adjustedTable21 = table10;
adjustedTable21.light_factor = light_factor;

adjustedTable22 = table10;
adjustedTable22.CO_factor = CO_factor;

adjustedTable23 = table10;
adjustedTable23.basicalconfiguration_factor = basicalconfiguration_factor;

adjustedTable24 = table10;
adjustedTable24.inspection_factor = inspection_factor;

adjustedTable25 = table10;
adjustedTable25.risk_factor = risk_factor;

adjustedTable26 = table10;
adjustedTable26.noise_factor = noise_factor;

adjustedTable27 = table10;
adjustedTable27.TVOC_factor = TVOC_factor;

adjustedTable28 = table10;
adjustedTable28.wind_factor = wind_factor;

adjustedTable29 = table10;
adjustedTable29.humidity_factor = humidity_factor;

adjustedTable30 = table10;
adjustedTable30.emergencylocation_factor = emergencylocation_factor;



%放在同一表下
table11 = vertcat(adjustedTable1, adjustedTable2,adjustedTable3,adjustedTable4,adjustedTable5,adjustedTable6,adjustedTable7,adjustedTable8,adjustedTable9,adjustedTable10,adjustedTable11, adjustedTable12,adjustedTable13,adjustedTable14,adjustedTable15,adjustedTable16,adjustedTable17,adjustedTable18,adjustedTable19,adjustedTable20,adjustedTable21, adjustedTable22,adjustedTable23,adjustedTable24,adjustedTable25,adjustedTable26,adjustedTable27,adjustedTable28,adjustedTable29,adjustedTable30);



%删除重复项
[~, uniqueIndices] = unique(table11, 'rows', 'stable');
table11Unique = table11(uniqueIndices, :);

rows_table10Unique_noDup = table10{:,:};
rows_table11Unique = table11Unique{:,:};
[~, idx1] = ismember(rows_table11Unique, rows_table10Unique_noDup, 'rows');
table11 = table11Unique(~idx1, :);

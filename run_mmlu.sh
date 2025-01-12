#!/bin/bash

tasks=(
abstract_algebra 
anatomy
astronomy 
business_ethics 
clinical_knowledge 
college_biology 
college_chemistry 
college_computer_science 
college_mathematics 
college_medicine 
college_physics 
computer_security 
conceptual_physics 
econometrics 
electrical_engineering 
elementary_mathematics 
formal_logic
global_facts 
high_school_biology 
high_school_chemistry
high_school_computer_science 
high_school_european_history 
high_school_geography 
high_school_government_and_politics 
high_school_macroeconomics 
high_school_mathematics 
high_school_microeconomics 
high_school_physics 
high_school_psychology 
high_school_statistics 
high_school_us_history 
high_school_world_history 
human_aging
human_sexuality
international_law 
jurisprudence 
logical_fallacies 
machine_learning 
management 
marketing 
medical_genetics 
miscellaneous 
moral_disputes 
moral_scenarios 
nutrition 
philosophy 
prehistory 
professional_accounting 
professional_law 
professional_medicine 
professional_psychology 
public_relations 
security_studies 
sociology 
us_foreign_policy 
virology 
world_religions
)

for task in "${tasks[@]}"; do
    python src/evaluate.py -in "/home/wth/few_vs_zero/data/mmlu/token0/$task/test_0.pkl" -mask "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/random13.pth" -x 1.7 -d 0 &
    python src/evaluate.py -in "/home/wth/few_vs_zero/data/mmlu/token0/$task/test_0.pkl" -mask "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/random42.pth" -x 1.7 -d 1 &
    python src/evaluate.py -in "/home/wth/few_vs_zero/data/mmlu/token0/$task/test_0.pkl" -mask "/home/wth/few_vs_zero/data/gsm/activation_mask/Random/random1.pth" -x 1.7 -d 2 &
    wait
done
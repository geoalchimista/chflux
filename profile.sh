python3 -m cProfile -o flux_calc.prof flux_calc.py -c "user_config.yaml"
python3 -m pstats flux_calc.prof

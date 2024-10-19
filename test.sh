# age
# {'unknown', 'old', 'nonOld'}
python environment.py --mode single_pipeline --target old --domain age
python environment.py --mode single_pipeline --target nonold --domain age

# ===========================================
# disability_status
# {'unknown', 'nonDisabled', 'disabled'}
python environment.py --mode single_pipeline --target disabled --domain disability_status
python environment.py --mode single_pipeline --target nonDisabled --domain disability_status

# ===========================================
# gender_identity
# {'girl', 'nonTrans_M', 'trans_M', 'nonTrans', 'nonTrans_F', 'man', 'M', 'trans_F', 'trans', 'F', 'unknown', 'woman', 'boy'}
python environment.py --mode single_pipeline --target trans --domain gender_identity
python environment.py --mode single_pipeline --target nonTrans --domain gender_identity
python environment.py --mode single_pipeline --target trans_M --domain gender_identity
python environment.py --mode single_pipeline --target trans_F --domain gender_identity
python environment.py --mode single_pipeline --target nonTrans_F --domain gender_identity
python environment.py --mode single_pipeline --target boy --domain gender_identity
python environment.py --mode single_pipeline --target girl --domain gender_identity
python environment.py --mode single_pipeline --target man --domain gender_identity
python environment.py --mode single_pipeline --target woman --domain gender_identity

# ===========================================
# nationality
# {'AsiaPacific', 'NorthAmerica', 'MiddleEast', 'ArabStates', 'LatinSouthAmerica', 'Africa', 'Europe', 'unknown'}
python environment.py --mode single_pipeline --target AsiaPacific --domain nationality
python environment.py --mode single_pipeline --target NorthAmerica --domain nationality
python environment.py --mode single_pipeline --target MiddleEast --domain nationality
python environment.py --mode single_pipeline --target ArabStates --domain nationality
python environment.py --mode single_pipeline --target LatinSouthAmerica --domain nationality
python environment.py --mode single_pipeline --target Africa --domain nationality
python environment.py --mode single_pipeline --target Europe --domain nationality

# ===========================================
# physical_appearance
# {'noVisibleDifference', 'notPregnant', 'posDress', 'obese', 'tall', 'negDress', 'pregnant', 'nonObese', 'visibleDifference', 'unknown', 'short'}
python environment.py --mode single_pipeline --target noVisibleDifference --domain physical_appearance
python environment.py --mode single_pipeline --target notPregnant --domain physical_appearance
python environment.py --mode single_pipeline --target posDress --domain physical_appearance
python environment.py --mode single_pipeline --target obese --domain physical_appearance
python environment.py --mode single_pipeline --target tall --domain physical_appearance
python environment.py --mode single_pipeline --target negDress --domain physical_appearance
python environment.py --mode single_pipeline --target pregnant --domain physical_appearance
python environment.py --mode single_pipeline --target nonObese --domain physical_appearance
python environment.py --mode single_pipeline --target visibleDifference --domain physical_appearance
python environment.py --mode single_pipeline --target unknown --domain physical_appearance
python environment.py --mode single_pipeline --target short --domain physical_appearance

# ===========================================
# religion
# {'Hindu', 'Sikh', 'Muslim', 'Mormon', 'Jewish', 'Catholic', 'Protestant', 'Atheist', 'unknown', 'Buddhist', 'Christian'}
python environment.py --mode single_pipeline --target Hindu --domain religion
python environment.py --mode single_pipeline --target Sikh --domain religion
python environment.py --mode single_pipeline --target Muslim --domain religion
python environment.py --mode single_pipeline --target Mormon --domain religion
python environment.py --mode single_pipeline --target Jewish --domain religion
python environment.py --mode single_pipeline --target Catholic --domain religion
python environment.py --mode single_pipeline --target Protestant --domain religion
python environment.py --mode single_pipeline --target Atheist --domain religion
python environment.py --mode single_pipeline --target Buddhist --domain religion
python environment.py --mode single_pipeline --target Christian --domain religion

# ===========================================
# ses
# {'lowSES', 'highSES', 'unknown'}
python environment.py --mode single_pipeline --target lowSES --domain ses
python environment.py --mode single_pipeline --target highSES --domain ses

# ===========================================
# sexual_orientation
# {'gay', 'pansexual', 'lesbian', 'straight', 'bisexual', 'unknown'}
python environment.py --mode single_pipeline --target gay --domain sexual_orientation
python environment.py --mode single_pipeline --target pansexual --domain sexual_orientation
python environment.py --mode single_pipeline --target lesbian --domain sexual_orientation
python environment.py --mode single_pipeline --target straight --domain sexual_orientation
python environment.py --mode single_pipeline --target bisexual --domain sexual_orientation
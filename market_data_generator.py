import random

# a seniority junior[0,0.11,0.22,0.33] medium [0.44,0.55,0.66] senior[0.77,0.88,1]
# b purchase no_plan [0,0.11,0.22,0.33] P2[0.44,0.55,0.66] P3[0.77,0.88,1]
# c company size small[<0.34] medium[<0.66] large [>0.66]
# d contact  nothing[0,0.11,0.22,0.33] brochure[0.44,0.55,0.66] email[0.77,0.88,1]
# e action nothing[0] doing[1]

def get_action_value(a, b, c, d):
    if a < 0.37 and b > 0.7 and c > 0.37 and d < 0.71 and d > 0.37:
        e = 0.5
    elif a > 0.37 and a < 0.71 and c > 0.37 and d < 0.71 and d > 0.37:
        e = 0.5
    elif a > 0.37 and a < 0.71 and b > 0.7 and d < 0.71 and d > 0.37:  # and (a<0.67 or c<0.67 or b<0.5):
        e = 0.5
    elif a > 0.7 and d < 0.71 and d > 0.37:  # and (a<0.67 or c<0.67 or b<0.5):
        e = 0.5
    elif a < 0.37 and b > 0.7 and c > 0.37 and d > 0.7:
        e = 1
    elif a > 0.37 and a < 0.7 and d > 0.7:
        e = 1
    elif a > 0.7 and d > 0.7:  # and (a<0.67 or c<0.67 or b<0.5):
        e = 1
    else:
        e = 0
    return e


def get_float(int_value, decimals=2):
    return round(int_value / 9.0, decimals)


title_str = "Seniority,Purchase_Propensity,CompanySize,Contactable,Action\n"
format_str = "%s,%s,%s,%s,%s\n"

int_data_file = open("data_value_int.csv", "w+")
int_data_file.write(title_str)

float_data_file = open("data_value_float_bk.csv", "w+")
float_data_file.write(title_str)

for i in range(5000):
    seniority_int = int(10 * random.random())
    seniority_float = get_float(seniority_int)

    purchase_propensity_int = int(10 * random.random())
    purchase_propensity_float = get_float(purchase_propensity_int)

    company_size_int = int(10* random.random())
    company_size_float = get_float(company_size_int)

    contactable_int = int(10 * random.random())
    contactable_float = get_float(contactable_int)

    action = get_action_value(seniority_float, purchase_propensity_float, company_size_float, contactable_float)

    int_data_file.write( format_str % (seniority_int, purchase_propensity_int, company_size_int, contactable_int, action))
    float_data_file.write(format_str % (seniority_float, purchase_propensity_float, company_size_float, contactable_float, action))

int_data_file.flush()
int_data_file.close()

float_data_file.flush()
float_data_file.close()

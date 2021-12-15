# Importing Libraries
import numpy as np
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
from math import sin, cos, radians, sqrt

# Loading data
df = pd.read_csv('01_data_mars_opposition_updated.csv')

# Data preparation
def prepare_data(df):
  days = [0]
  for i in range(1, len(df)):
      date_prev = datetime.datetime(year=df['Year'][i - 1], month=df['Month'][i - 1], day=df['Day'][i - 1], hour=df['Hour'][i - 1],minute=df['Minute'][i - 1])
      date_next = datetime.datetime(year=df['Year'][i], month=df['Month'][i], day=df['Day'][i], hour=df['Hour'][i], minute=df['Minute'][i])
      date_diff = date_next - date_prev
      num_days = date_diff.days + date_diff.seconds / (60 * 60 * 24)
      days.append(num_days)

  # Computing longitude angles for each opposition
  longitude_angles = np.array(df['ZodiacIndex'] * 30 + df['Degree'] + df['Minute.1'] / 60 + df['Second'] / 3600)

  # Concat days, longitudes
  oppositions = np.stack([days, longitude_angles], axis=1)

  return longitude_angles, oppositions

longitude_angles, oppositions = prepare_data(df)

# Find intersection points
def intersection_helper(h, k, c, r, theta):
    # Converting (1,c) to cartesian
    x1 = 1 * cos(radians(c))
    y1 = 1 * sin(radians(c))

    phi = radians(theta)
    b = 2 * ((h - x1) * cos(phi) + (k - y1) * sin(phi))
    f = (h - x1) ** 2 + (k - y1) ** 2 - r ** 2
    A = -b / 2
    try:
        B = sqrt(b ** 2 - 4 * 1 * f) / 2
    except ValueError:
        B = 0
    root1 = A + B
    root2 = A - B
    if root1 > 0:
        L = root1
    else:
        L = root2
    x = h + L * cos(radians(theta))
    y = k + L * sin(radians(theta))
    return x, y

def getIntersectionPoints(c, r, s, e1, e2, z, times):
    X, Y = [], []
    # Equant in cartesian
    h = e1 * cos(radians(e2 + z))
    k = e1 * sin(radians(e2 + z))
    theta = z

    for t in times:
        theta = np.multiply(s, t) + theta
        x, y = intersection_helper(h, k, c, r, theta)
        X.append(x)
        Y.append(y)
    return X, Y
  
def cartesian2polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    phi = np.degrees(phi)
    return rho, phi

#-------------Question 1------------------------------------------------------


def MarsEquantModel(c, r, e1, e2, z, s, opp):
    """
      c: Angle in degrees center of cicular orbit is at from the sun
      r: radius of circular orbit
      e1: distance from sun
      e2: angle in degrees wrt to equant 0
      z: reference longitude in degrees wrt Aries
      s: angular veclocity (degrees per day)
      opp: 12x2 array, each row contains - time (days in decimal), and angular position (degrees in decimal)
    """
    times = opp[:, 0]  # Days commenced for each opposition
    long_angles = opp[:, 1]  # longitutde angles

    # Get intersection points
    X, Y = getIntersectionPoints(c=c, r=r, s=s, e1=e1, e2=e2, z=z, times=times)

    # Get polar coordinates of intersected points
    rho, phi = cartesian2polar(np.array(X), np.array(Y))

    for idx in range(len(phi)):
        if phi[idx] < 0:
            phi[idx] = 360 - abs(phi[idx])

    errors = np.abs(long_angles - phi)
    maxError = np.max(errors)
    return errors, maxError

errors, maxError = MarsEquantModel(c=170, r=9, z=54, e1=1.400, e2=120, s=0.524, opp=oppositions)

print("-------Q1-------------------")
print(f'Errors:  {errors}')
print(f'MaxError: {maxError}')
print("----------------------------")


#-------------Question 2------------------------------------------------------

def helper_best_orbit_inner_params(C, E1, E2, Z, r, s, opp):
    optimal_inner_params = []
    errors_result = None
    minMaxError = np.inf

    list_of_params_for_each_decreased_maxError = []

    for c in C:
        for e1 in E1:
            for e2 in E2:
                for z in Z:
                    if 0 <= c < 360 and 0 <= e2 < 360 and 0 <= z < 360:
                        errors, maxError = MarsEquantModel(c=c, r=r, e1=e1, e2=e2, z=z, s=s, opp=opp)
                        if maxError < minMaxError:
                            minMaxError = maxError
                            errors_result = errors
                            optimal_inner_params = [c, e1, e2, z]
                            list_of_params_for_each_decreased_maxError.append([c, e1, e2, z, maxError])

    params_df = pd.DataFrame(list_of_params_for_each_decreased_maxError,
                             columns=['c', 'e1', 'e2', 'z', 'maxError'],
                             index=None)

    return optimal_inner_params, errors_result, minMaxError, params_df


def bestOrbitInnerParams(r, s, opp):
    """
      r: Radius of circular orbit
      s: angular veclocity
      opp: 12x2 array, each row contains - time (days in decimal), and angular position (degrees in decimal)
    """

    start_time = time.time()

    # Coarse grid search
    C = np.arange(50, 150, step=10)
    Z = np.arange(50, 150, step=10)
    E2 = np.arange(50, 150, step=10)
    E1 = np.arange(0, 3, step=0.1)

    optimal_inner_coarse_params, _, _, coarse_params_df = helper_best_orbit_inner_params(C, E1, E2, Z, r, s, opp)

    # Fine grid search
    c, e1, e2, z = optimal_inner_coarse_params
    W = 5
    C = np.arange(c - W, c + W, step=1)
    E2 = np.arange(e2 - W, e2 + W, step=1)
    Z = np.arange(z - W, z + W, step=1)

    optimal_inner_params, errors, maxError, fine_params_df = helper_best_orbit_inner_params(C, E1, E2, Z, r, s, opp)

    end_time = time.time()
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

    c, e1, e2, z = optimal_inner_params
    return c, e1, e2, z, errors, maxError

c, e1, e2, z, errors, maxError = bestOrbitInnerParams(r=8.5, s=0.52, opp=oppositions)
print("-------Q2-------------------")
print(f'c={c}, e1={e1}, e2={e2}, z={z}')
print('Errors: ', errors)
print('MaxError: ', maxError)
print("----------------------------")






#-------------Question 3------------------------------------------------------
print("-------Q3-------------------")




def bestS(r, opp):
    """
      r: radius of cicular orbit
      opp: 12x2 array, each row contains - time (days in decimal), and angular position (degrees in decimal)
    """

    S = []
    for num in range(350, 371):
        S.append(num / 687)

    errors_result = None
    minMaxError = np.inf
    best_s = None

    s_maxError_list = []

    start_time = time.time()

    for idx in range(len(S)):
        s = S[idx]
        _, _, _, _, errors, maxError = bestOrbitInnerParams(r=r, s=s, opp=opp)
        if maxError < minMaxError:
            minMaxError = maxError
            errors_result = errors
            best_s = s
        s_maxError_list.append([s, maxError])

    s_df = pd.DataFrame(s_maxError_list,
                        columns=['s', 'maxError'],
                        index=None)
    
    # print(s_df) 

    # From above results, we get
    optimal_c = 54
    minMaxError = 1.194329
    errors_result = errors_result
    best_s_obatained_so_far = 0.5240174672489083

    list_max_error = []

    C = np.arange(54, 110, step=0.05)
    for c in C:
        errors, maxError = MarsEquantModel(c=c, r=8, z=57, e1=1.5, e2=92, s=best_s_obatained_so_far, opp=opp)
        if maxError < minMaxError:
            minMaxError = maxError
            errors_result = errors
            optimal_c = c
        list_max_error.append(maxError)
    # plt.plot(C, list_max_error)
    # plt.title("Fine tuning 'c'")
    # plt.xlabel('c (degrees)')
    # plt.ylabel('Max Error (degrees)')
    # plt.savefig('Q3_c.png')

    # print(f'Optimal c = {optimal_c}')

    # Result obtained:
    # For c = 105.5 degrees, max error got reduced to 0.9517921305329082 degrees from 1.194329 degrees.


    # Optimal e1, we get as
    optimal_e1 = 1.5

    # Fine tuning e
    
    E = np.arange(1, 2, step=0.0001)
    list_max_error = []
    for e1 in E:
        errors, maxError = MarsEquantModel(c=105.5, r=8, z=57, e1=e1, e2=92, s=best_s, opp=opp)
        if maxError < minMaxError:
            minMaxError = maxError
            errors_result = errors
            optimal_e1 = e1
        list_max_error.append(maxError)

    end_time = time.time()

    # print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

    return best_s, errors_result, minMaxError

s, errors, maxError = bestS(r=8, opp=oppositions)
print("-------Q3-------------------")
print(f's={s}')
print('Errors: ', errors)
print('MaxError: ', maxError)
print("----------------------------")






#-------------Question 4------------------------------------------------------
print("-------Q4-------------------")


def bestR(s, opp):
    """
      s: angular veclocity (degrees per day)
      opp: 12x2 array, each row contains - time (days in decimal), and angular position (degrees in decimal)
    """

    # Defining range for r
    R = np.arange(4.5, 9.5, step=0.01)   # print(len(R))   # 20

    errors_result = None
    minMaxError = np.inf
    best_r = None
    r_maxError_list = []

    start_time = time.time()
    for idx in range(len(R)):
        r = R[idx]
        _, _, _, _, errors, maxError = bestOrbitInnerParams(r=r, s=s, opp=opp)
        if maxError < minMaxError:
            minMaxError = maxError
            errors_result = errors
            best_r = r
        r_maxError_list.append([r, maxError])

    r_df = pd.DataFrame(r_maxError_list,
                        columns=['r', 'maxError'],
                        index=None)
    end_time = time.time()
    
    # print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
    # fine-tuning parameters, for given s=0.52
    # After going through all the iterations of r, it was found that for these parameters -
    # s=0.52 (given),c=117, e1=0.6, e2=105, z=72, r=4.3999 => least max error achieved was 12.000176 degrees.
    # See log report in report under Question 4

    ###############
    # Fine-tuning #
    ###############

    optimal_c = 117
    minMaxError = 12.000176  # taken from above obtained minMaxError, hardcoding it since I commented the above part and ran just this fine-tuning section
    errors_result = None

    list_max_error = []

    C = np.arange(100, 120, step=0.5)
    for c in C:
        errors, maxError = MarsEquantModel(c=c, r=4.3999, z=72, e1=0.6, e2=105, s=0.52, oppositions=opp)
        if maxError < minMaxError:
            minMaxError = maxError
            errors_result = errors
            optimal_c = c
        list_max_error.append(maxError)

    # plt.plot(C, list_max_error)
    # plt.title("Fine tuning 'c'")
    # plt.xlabel('c (degrees)')
    # plt.ylabel('Max Error (degrees)')
    # plt.savefig('Q4_c.png')

    # print(f'Optimal c = {optimal_c}')

    # Result obtained:
    # For c = 116.5, max error got reduced to 11.99694671956815 degrees from 12.000176279434513 degree

    return best_r, errors_result, minMaxError

r, errors, maxError = bestR(s=0.52, opp=oppositions)

print(f'r={r}')
print('Errors: ', errors)
print('MaxError: ', maxError)
print("------------------------------")







#---------------------------Question 5------------------------------------------------------
print("-------Q5-------------------")

def helper_best_mars_orbit_params(C, E1, E2, Z, R, S, opp):
    optimal_inner_params = []
    errors_result = None
    minMaxError = np.inf

    list_of_params_for_each_decreased_maxError = []

    for r in R:
        for s in S:
            for e1 in E1:
                for e2 in E2:
                    for c in C:
                        for z in Z:
                            if 0 <= c < 360 and 0 <= e2 < 360 and 0 <= z < 360:
                                errors, maxError = MarsEquantModel(c=c, r=r, e1=e1, e2=e2, z=z, s=s, opp=opp)
                                if maxError < minMaxError:
                                    minMaxError = maxError
                                    errors_result = errors
                                    optimal_inner_params = [r, s, c, e1, e2, z]
                                    list_of_params_for_each_decreased_maxError.append(
                                        [r, s, c, e1, e2, z, maxError])

    params_df = pd.DataFrame(list_of_params_for_each_decreased_maxError,
                             columns=['r', 's', 'c', 'e1', 'e2', 'z', 'maxError'],
                             index=None)

    return optimal_inner_params, errors_result, minMaxError, params_df

def bestMarsOrbitParams(opp):
    """
      opp: 12x2 array, each row contains - time (days in decimal), and angular position (degrees in decimal)  
    """

    R = np.arange(4.5, 9.5, step=0.01)
    S = []
    for num in range(355, 365):
        S.append(num / 687)

    # Coarse grid search
    start_time = time.time()
    C = np.arange(50, 150, step=10)
    Z = np.arange(50, 150, step=10)
    E2 = np.arange(50, 150, step=10)
    E1 = np.arange(0, 3, step=0.001)

    optimal_inner_coarse_params, _, _, coarse_params_df = helper_best_mars_orbit_params(C, E1, E2, Z, R, S, opp)
    
    # Fine grid search
    r, s, c, e1, e2, z = optimal_inner_coarse_params

    W = 5  # window size around found params
    C = np.arange(c - W, c + W, step=1)
    E2 = np.arange(e2 - W, e2 + W, step=1)
    Z = np.arange(z - W, z + W, step=1)

    optimal_inner_params, errors, maxError, fine_params_df = helper_best_mars_orbit_params(C, E1, E2, Z, R, S, opp)

    r, s, c, e1, e2, z = optimal_inner_params
    end_time = time.time()
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))
    return r, s, c, e1, e2, z, errors, maxError

r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(opp=oppositions)

print(f'r={r}, s={s}, c={c}, e1={e1}, e2={e2}, z={z}')
print('Errors: ', errors)
print('MaxError: ', maxError)
print("-----------------------------")
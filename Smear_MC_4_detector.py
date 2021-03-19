# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:08:14 2021

@author: laure
"""

ath = r"C:\Users\laure\Documents\Physics\Year 3\Group Study\Point_Source-Truth_Data_1.csv"
# path = 'D:/University/Year 3/Group Studies/Data/Lab Data/HGTD_23_02_NEW_ENERGY UPPERBOUND.csv'
dataframe = pd.read_csv(path)

# dataframe.loc[dataframe["Energy (keV)_1"] > 145.2, "Energy (keV)_1"] = np.nan

# absorber_2 = [30.961, 0, -23.923]
# dataframe.loc[dataframe["X_2 [cm]"] > 26, "X_2 [cm]"] = absorber_2[0]
# dataframe.loc[dataframe["X_2 [cm]"] > 26, "Y_2 [cm]"] = absorber_2[1]
# dataframe.loc[dataframe["X_2 [cm]"] > 26, "Z_2 [cm]"] = absorber_2[2]

# absorber_3 = [-30.961, 0, -23.923]
# dataframe.loc[dataframe["X_2 [cm]"] < -25, "X_2 [cm]"] = absorber_3[0]
# dataframe.loc[dataframe["X_2 [cm]"] < -25, "Y_2 [cm]"] = absorber_3[1]
# dataframe.loc[dataframe["X_2 [cm]"] < -25, "Z_2 [cm]"] = absorber_3[2]

# absorber_4 = [16.397, 0, -31.954]
# dataframe.loc[(dataframe["X_2 [cm]"] > 10) & (dataframe["X_2 [cm]"] < 26), "X_2 [cm]"] = absorber_4[0]
# dataframe.loc[(dataframe["X_2 [cm]"] > 10) & (dataframe["X_2 [cm]"] < 26), "Y_2 [cm]"] = absorber_4[1]
# dataframe.loc[(dataframe["X_2 [cm]"] > 10) & (dataframe["X_2 [cm]"] < 26), "Z_2 [cm]"] = absorber_4[2]

# absorber_5 = [-16.397, 0, -31.954]
# dataframe.loc[(dataframe["X_2 [cm]"] < -10) & (dataframe["X_2 [cm]"] > -25), "X_2 [cm]"] = absorber_5[0]
# dataframe.loc[(dataframe["X_2 [cm]"] < -10) & (dataframe["X_2 [cm]"] > -25), "Y_2 [cm]"] = absorber_5[1]
# dataframe.loc[(dataframe["X_2 [cm]"] < -10) & (dataframe["X_2 [cm]"] > -25), "Z_2 [cm]"] = absorber_5[2]

def func(x, a, b, c):
    #return a*np.exp((-(x-b)**2)/(2*c**2))
    return -(a*(x**2) + b*x + c)
    #return a*np.abs(x) + b

def z_slice_selector(z_min, z_max, z_slices):
    max_pixel_value = []
    z_value = []
    for i in np.linspace(z_min, z_max, z_slices):
        max_pv = get_image([points1], 10, [i, i], 662E3, 100, R=[0,0], ROI=[-100, 100, -100, 100], steps=[50,50], ZoomOut=0, plot_individuals=True)[2]
        max_pixel_value.append(max_pv)
        z_value.append(i)
        
        
    plt.scatter(z_value, max_pixel_value)
    
    popt, pcov = curve_fit(func, np.array(z_value), np.array(max_pixel_value))
    slice_selected = -popt[1]/(2*popt[0])
    print(f'parameters = {popt}')
    plt.plot(np.array(z_value), func(np.array(z_value), *popt), 'r-',
         label=f'Correct image slice distance = {round(slice_selected, 3)} cm')
    plt.title('Maximum intensity vs image slice distance z')
    plt.ylabel('Maximum intensity')
    plt.xlabel('Image slice z distance (cm)')
    plt.legend()
    plt.show()
    
    #print(f'max_pixel, z_value = {max_pixel_value, z_value}')
    return z_value, max_pixel_value
        
z_value, max_pixel_value = z_slice_selector(19, 31, 10)
print(f'z_value, max_pixel_value = {z_value, max_pixel_value}')
print("--- %s seconds ---" % (time.time() - start_time))
import csv
import datetime 
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import datetime
from sklearn import linear_model
from dateutil.relativedelta import relativedelta

def get_gps(gps_dir, gps_list_file, plot_box, start_date, end_date, unit, key_length):

    inf=open(gps_list_file)
    next(inf)
    reader=csv.reader(inf, delimiter=' ')
    zipper=zip(*reader)
    gpslist = list(next(zipper))
    latlist = list(next(zipper))
    lonlist = list(next(zipper))
    new_gpslist = []
    new_latlist = []
    new_lonlist = []
    
    for i in range(0, len(gpslist)):
       gps = gpslist[i]
       lat = latlist[i]
       lon = lonlist[i]
       if float(lat) >= plot_box[0] and float(lat) <= plot_box[1] and float(lon) >= plot_box[2] and float(lon) <= plot_box[3]:
            new_gpslist.append(gps)
            new_latlist.append(lat)
            new_lonlist.append(lon)

    lat,lon,U,V,Z = get_quiver(gps_dir, new_gpslist, new_lonlist, new_latlist, start_date, end_date);
    duration_years, quiver_label = generate_quiver_label(unit, key_length, start_date, end_date)
    
    if unit == 'cm':
        U = [u * duration_years for u in U]
        V = [v * duration_years for v in V]
        Z = [z * duration_years for z in Z]

    return new_gpslist, lat, lon, U, V, Z, quiver_label

def get_gps_vel(gps_dir, sitename,time1,time2):
    filename =  gps_dir + '/' + sitename + '.txt'
    dfin = read_csv(filename, header=0, delimiter=r"\s+")
    index = ['Time', 'East', 'North', 'Up']
    dataval=DataFrame(index=index);
    dataerr=DataFrame(index=index);
    dataval=concat([dfin['YYMMMDD'].rename('date'), (dfin['_e0(m)']+dfin['__east(m)']).rename('east'), (dfin['____n0(m)']+dfin['_north(m)']).rename('north'), 
                    (dfin['u0(m)']+dfin['____up(m)']).rename('up'),dfin['yyyy.yyyy'].rename('dateval')], axis=1)
    dataerr=concat([dfin['yyyy.yyyy'].rename('date'), dfin['sig_e(m)'], dfin['sig_n(m)'], dfin['sig_u(m)']], axis=1, 
                 ignore_index=False);
    dataval['date']=pd.to_datetime(dataval['date'], format='%y%b%d', errors='ignore')
    dataerr['date']=pd.to_datetime(dataval['date'], format='%y%b%d', errors='ignore')
    time1 = pd.to_datetime(time1)
    time2 = pd.to_datetime(time2)
    mask= (dataval['date'] > time1) & (dataval['date'] < time2)
    dataval=dataval[mask];dataerr=dataerr[mask];
    regr = linear_model.LinearRegression()
    regr.fit(dataval['dateval'].values.reshape(-1,1),dataval['east'].values.reshape(-1,1));east_vel=regr.coef_[0][0];
    regr.fit(dataval['dateval'].values.reshape(-1,1),dataval['north'].values.reshape(-1,1));north_vel=regr.coef_[0][0];
    regr.fit(dataval['dateval'].values.reshape(-1,1),dataval['up'].values.reshape(-1,1));up_vel=regr.coef_[0][0];
    return east_vel*1000, north_vel*1000,up_vel*1000;

def get_quiver(gps_dir, gpslist,lonlist,latlist,start_date,end_date):    
    date1 = datetime.strptime(start_date, "%Y%m%d") 
    date2 = datetime.strptime(end_date, "%Y%m%d")
    u_ref, v_ref, z_ref = get_gps_vel(gps_dir, 'MKEA', date1, date2)  #print u_ref,v_ref,z_ref
    X,Y,U,V,Z=[],[],[],[],[]      
    for i in range(len(gpslist)):
        try:
            u,v,z=get_gps_vel(gps_dir, gpslist[i],date1,date2);u=u-u_ref;v=v-v_ref;
            U.append(float(u));V.append(float(v));Z.append(float(z));
            X.append(float(lonlist[i])),Y.append(float(latlist[i]));
        except:
            pass
    return X,Y,U,V,Z

def generate_quiver_label(unit, key_length, start_date, end_date):
    if unit == 'cm/yr':
        duration_years = 1
    if unit == 'cm':
        date1 = datetime.strptime(start_date, "%Y%m%d") 
        date2 = datetime.strptime(end_date, "%Y%m%d")
        duration = relativedelta(date2, date1)
        total_days = duration.days + (duration.months * 30) + (duration.years * 365)
        duration_years=total_days/365

    str_label = f"{key_length} {unit}"
    return duration_years, str_label

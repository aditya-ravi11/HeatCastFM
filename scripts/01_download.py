"""Download IMD gridded Tmax/Tmin and the Open-Meteo station proxy."""
from heatcast import data_imd, data_openmeteo

if __name__ == "__main__":
    data_imd.download()
    data_openmeteo.build()

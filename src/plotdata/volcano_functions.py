import os
import json
from datetime import datetime
import pandas as pd
import requests


JSON_DOWNLOAD_URL = 'https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wms?service=WFS&version=1.0.0&request=GetFeature&typeName=GVP-VOTW:E3WebApp_Eruptions1960&outputFormat=application%2Fjson'

# TODO to replace elninos with the following API #
# TODO eventually move to helper_functions.py
if False:
    # CHECK THIS FIRST https://psl.noaa.gov/enso/mei/
    req = requests.get('https://psl.noaa.gov/enso/mei/data/meiv2.data')
    print(req.text)

###################################################


def get_volcano_json(jsonfile, url):
    """
    Retrieves volcano data from a JSON file or a remote URL.

    Args:
        jsonfile (str): The path to the local JSON file.
        url (str): The URL to retrieve the JSON data from.

    Returns:
        dict: The JSON data containing volcano information.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as err:
        print("Error: ", err)
        print("Loading from local file")

        if not os.path.exists(jsonfile):
            download_volcano_json(jsonfile, JSON_DOWNLOAD_URL)

        f = open(jsonfile)
        data = json.load(f)

    return data


def volcanoes_list(jsonfile):
    """
    Retrieves a list of volcano names from a JSON file.

    Args:
        jsonfile (str): The path to the JSON file containing volcano data.

    Returns:
        None
    """
    data = get_volcano_json(jsonfile, JSON_DOWNLOAD_URL)

    volcanoName = []
    volcanoId = []
    volcanoCoordinates = []

    for j in data['features']:
        if j['properties']['VolcanoNumber'] not in volcanoId:
            volcanoName.append(j['properties']['VolcanoName'])
            volcanoId.append(j['properties']['VolcanoNumber'])
            volcanoCoordinates.append(j['geometry']['coordinates'])


    for volcano, id, coord in zip(volcanoName, volcanoId, volcanoCoordinates):
        print(f'{volcano}, id: {id}, coordinates: {coord[0]}, {coord[1]}')

    return volcanoName


def get_volcano_coord_id(jsonfile, volcanoName: str):
    data = get_volcano_json(jsonfile, JSON_DOWNLOAD_URL)

    for j in data['features']:
        if j['properties']['VolcanoName'] == volcanoName:
            id = j['properties']['VolcanoNumber']

            coordinates = j['geometry']['coordinates']
            coordinates = coordinates[::-1]

            return coordinates, id


def get_volcano_coord_name(jsonfile, volcanoId):
    data = get_volcano_json(jsonfile, JSON_DOWNLOAD_URL)

    for j in data['features']:
        if j['properties']['VolcanoNumber'] == int(volcanoId):
            name = j['properties']['VolcanoNumber']

            coordinates = j['geometry']['coordinates']
            coordinates = coordinates[::-1]

            return coordinates, name


def get_volcano_event(jsonfile, volcanoName: str, start_date, end_date, strength = 0):
    """
    Extracts information about a specific volcano from a JSON file.

    Args:
        jsonfile (str): The path to the JSON file containing volcano data.
        volcanoName (str): The name of the volcano to extract information for.

    Returns:
        tuple: A tuple containing the start dates of eruptions, a date list, and the coordinates of the volcano.
    """
    volcano = {f"{volcanoName}": {},}
    column_names = ['Start', 'End', 'Max Explosivity']
    frame_data = []
    name = ''

    data = get_volcano_json(jsonfile, JSON_DOWNLOAD_URL)

    first_day = datetime.strptime(start_date, '%Y%m%d').date() if isinstance(start_date, str) else start_date
    last_day = datetime.strptime(end_date, '%Y%m%d').date() if isinstance(end_date, str) else end_date

    strength = int(strength)

    # Iterate over the features in the data
    for j in data['features']:
        if j['properties']['VolcanoName'] == volcanoName:
            id = j['properties']['VolcanoNumber']
            name = (j['properties']['VolcanoName'])
            start = datetime.strptime((j['properties']['StartDate']), '%Y%m%d').date()

            try:
                end = datetime.strptime((j['properties']['EndDate']), '%Y%m%d').date()

            except:
                end = 'None'

            coordinates = j['geometry']['coordinates']

            volcano[name] = {"id": id,
                             "coordinates": coordinates[::-1]}

            print(f'{name} (id: {id}) eruption started {start} and ended {end}')

            # If the start date is within the date range
            if start >= first_day and start <= last_day:
                # start_dates.append(start)
                if j['properties']['ExplosivityIndexMax'] >= strength:
                    frame_data.append([start, end, j['properties']['ExplosivityIndexMax']])

    if name == '':
        raise ValueError(f'Volcano {volcanoName} not found, check for typos')

    if frame_data != []:
        df = pd.DataFrame(frame_data, columns=column_names)
        print('-'*50)
        print('Sorting eruptions by date...')
        print('-'*50)

        df.sort_values(by='Start', inplace=True)
        volcano[name]['eruptions'] = df

        for d in df['Start']:
            print('Extracted eruption in date: ', d)

        print('-'*50)

    print('')

    return volcano

# TODO see if xlsx file is needed
# def get_volcanoes():
#     """
#     Retrieves volcano data from an Excel file and returns a dictionary of volcano information.

#     Returns:
#         dict: A dictionary containing volcano information, with volcano names as keys and a dictionary of volcano attributes as values.
#             The volcano attributes include 'id', 'latitude', and 'longitude'.
#     """
#     df = pd.read_excel(VOLCANO_FILE, skiprows=1)
#     df = df[df['Precip'] != False]

#     volcano_dict = {
#         r['Volcano Name'] : {
#             'id': r['Volcano Number'],
#             'latitude': r['Latitude'],
#             'longitude': r['Longitude']
#         } for _, r in df.iterrows()}

#     return volcano_dict


def download_volcano_json(json_path, json_download_url=JSON_DOWNLOAD_URL):
    """
    Downloads a JSON file containing volcano eruption data from a specified URL and saves it to the given file path.

    Args:
        json_path (str): The file path where the JSON file will be saved.
        json_download_url (str): The URL from which the JSON file will be downloaded.

    Raises:
        requests.exceptions.HTTPError: If an HTTP error occurs while downloading the JSON file.

    Returns:
        None
    """
    try:
        result = requests.get(json_download_url)

        with open(json_path, 'wb') as f:
            f.write(result.content)

    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 404:
            msg = f'Error: {err.response.status_code} Url Not Found'
            raise ValueError(msg)

        else:
            msg = 'An HTTP error occurred: ' + str(err.response.status_code)
            raise ValueError(msg)


    if os.path.exists(json_path):
        print(f'Json file downloaded in {json_path}')

    else:
        print('Cannot create json file')
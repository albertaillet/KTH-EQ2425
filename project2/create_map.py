# %% Imports
import yaml
import requests
import plotly.graph_objects as go

FOLDER = 'data2'

# %% Load addresses from file
with open(f'{FOLDER}/addresses.yml') as f:
    addresses = yaml.load(f, Loader=yaml.SafeLoader)

# %% Getting latitude and longitude from OpenStreetMap API
url = 'https://nominatim.openstreetmap.org/search.php?'

def get_lat_lon(address: str) -> tuple:
    params = {
        'street': address,
        'city': 'Stockholm',
        'country': 'Sweden',
        'format': 'jsonv2',
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data[0]['lat'], data[0]['lon']

# %% Create a dictionary with the locations
locations = {}
for location_id, address in addresses.items():
    print('Getting location for {location_id}', end='\r')
    try:
        lat, lon = get_lat_lon(address)
        locations[location_id] = {
            'address': address,
            'lat': lat,
            'lon': lon,
        }
    except:
        print(f'Could not find {address} with id {location_id}')

# %% Save the locations to a yaml file
with open(f'{FOLDER}/locations.yml', 'w') as f:
    yaml.dump(locations, f, allow_unicode=True)

# %% Load the locations to a yaml file
with open(f'{FOLDER}/locations.yml') as f:
    locations = yaml.load(f, Loader=yaml.SafeLoader)

# %% Create a map with the locations
fig = go.Figure(go.Scattermapbox(
        lat=[l['lat'] for l in locations.values()],
        lon=[l['lon'] for l in locations.values()],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10
        ),
        text=['Location of Photos'],
    ),
    layout=go.Layout(
        hovermode='closest',
        mapbox_style='carto-positron',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=59.3393,
                lon=18.0686
            ),
            pitch=0,
            zoom=11
        ),
    )
)
fig
# %%
# %%


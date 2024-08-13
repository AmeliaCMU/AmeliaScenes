# Amelia-48 Dataset

Click [here](https://airlab-share-01.andrew.cmu.edu:9000/amelia-processed/amelia-10.zip) to download the **Amelia-48**.

Once downloaded, make sure to create a symbolic link of the `amelia` folder into the `datasets` folder of the repository.

The downloaded dataset follows this structure:
```bash
|-- amelia
    |-- assets
        | -- airport_icao
            | -- bkg_map.png
            | -- limits.json
            | -- airport_code_from_net.osm
        | ...
    |-- graph_data_a10v01os
        | -- airport_icao
            | -- semantic_graph.pkl
            | -- semantic_airport_icao.osm
            | -- semantic_graph.png
        | ...
    |-- traj_data_a10v08
        | -- airport_icao
            | -- AIRPORT_ICAO_<unix_timestamp>.csv
            | --
        | ...
```

### Assets

The `assets` folder has a subfolder for each airport (uses the airport's [ICAO](https://en.wikipedia.org/wiki/ICAO_airport_code)) containing the following:
* `bkg_map.png`: visual representation of the map, obtained using OpenStreetMap (OSM).
* `limits.json`: JSON file containing the Airport's extents.
* `airport_icao.osm`: the airport's map in OSM format.

### Graph (Map) Data

The `graph_data_a10v01os` folder has a subfolder for each airport containing semantic graphs representation obtained using [AmeliaMaps](https://github.com/AmeliaCMU/AmeliaMaps). Each sub-folder contains the following files:
* `semantic_graph.pkl`: contains the vectorized map graph with semantic attributes.
* `semantic_airport_icao.osm`: the semantic representation of the graph in OSM format
* `semantic_graph.png`: visual representation of the graph. Just shown for reference.

**NOTE** this folder contains the graphs for the 10 airports used in our training experiments. The full set of 48 maps is in the folder `graph_data_a48v01os`.

### Trajectory Data

The `traj_data_a10v08` folder has a subfolder for each airport containing the trajectory data in CSV format. Each file within an airport's subfolder represents an hour of data.

The files are named following the format ```AIRPORT_ICAO_<unix_timestamp>.csv```, and contains the following fields of data: ```Frame,ID,Altitude,Speed,Heading,Lat,Lon,Range,Bearing,Type,Interp,x,y```.
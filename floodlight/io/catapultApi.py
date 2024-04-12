import requests
import numpy as np
import floodlight

## Custom Errors
class APIRequestError(Exception):
    """Custom error class for API request failures."""
    def __init__(self, message):
        super().__init__(message)

## Utils
def _getAllplayerFormActivity_response(baseUrl,api_token, activity_id ):
    """
    baseUrl: url to the catapult api (example : "https://connect-eu.catapultsports.com/api/v6/")
    api_token:  authentication token to acess api
    
    returns:containing each player in the activity (response_data)

    """

    headers = {
    "accept": "application/json",
    "authorization": f"Bearer {api_token}"
    }
    
    endpoint = f"activities/{activity_id}/athletes/" 
    
    response = requests.get(baseUrl + endpoint, headers=headers)

    if response.status_code == 200:
        return response
    else: 
        raise APIRequestError(f"API request failed with status code {response.status_code}")
    


def getAllPlayersFromActivity(baseUrl,api_token, activity_id): 
    """
    
    """
    headers = {
    "accept": "application/json",
    "authorization": f"Bearer {api_token}"
    }
    allPlayersFromActivity_response = _getAllplayerFormActivity_response(baseUrl,api_token, activity_id )

    allPlayersFromActivity_JSON = allPlayersFromActivity_response.json()
    playerIDsInActivity_list = []
    for i in allPlayersFromActivity_JSON:
        playerIDsInActivity_list.append(i["id"])

    playerCoordList = [[] for _ in range(len(playerIDsInActivity_list))]
    response_data_json_list = []

    # get earliest timeStamp for Synchronisation of timeStamps
    earliestTimeStamp = np.inf
    latestTimeStamp = -np.inf
    for indx_i, i in enumerate(playerIDsInActivity_list):
    
        url = f'activities/{activity_id}/athletes/{i}/sensor'
        response_data_json_i = requests.get(baseUrl+ url, headers=headers)
        if response_data_json_i.status_code == 200:
            response_data_json_i = response_data_json_i.json()
            response_data_json_list.append(response_data_json_i)
            
            earliestTimeStamp_temp = response_data_json_i[0]["data"][0]["ts"]
            latestTimeStamp_temp = response_data_json_i[0]["data"][-1]["ts"]

            earliestTimeStamp = min(earliestTimeStamp_temp, earliestTimeStamp)
            latestTimeStamp = max (latestTimeStamp_temp, latestTimeStamp)

        else:
            raise APIRequestError(f"API request failed with status code {response_data_json_i.status_code}")

    #for indx_i , response_data_json_i in enumerate(response_data_json_list):
        firstTimeStamp = response_data_json_i[0]["data"][0]["ts"] 
        lastTimeStamp = response_data_json_i[0]["data"][-1]["ts"] 
        
        # make sure all the timestamps of the players are synchronised 
        for _ in range(firstTimeStamp - earliestTimeStamp_temp):
            for _ in range(10): #because 10hz sensor data
                playerCoordList[indx_i].append((np.nan,np.nan))

        for j in response_data_json_i[0]["data"]:
            x = j["x"]
            y = j["y"]
            playerCoordList[indx_i].append((x,y))
        
        # make sure all the timestamps of the players are synchronised 
        for _ in range(latestTimeStamp - lastTimeStamp):
            for _ in range(10): #because 10hz sensor data
                playerCoordList[indx_i].append((np.nan,np.nan))

    # reshape data to fit xy object format
    playerCoordList_transposedFor_XY = [[row[i] for row in playerCoordList] for i in range(len(playerCoordList[0]))] 
    xy_object_np = np.array(playerCoordList_transposedFor_XY)
    xy_object_np = xy_object_np.reshape(xy_object_np.shape[0], -1)

    framerate_value = 10  #because 10hz sensor data
    xy_object = floodlight.core.xy.XY(xy=xy_object_np, framerate=framerate_value)

    return xy_object